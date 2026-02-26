"""Algorithm 1 – Zero-Shot LLM-Driven ASR Decoder."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
import os
import csv
import time
import random

import numpy as np
import matplotlib.pyplot as plt
import torch

from config import Config
from acoustic_model import AcousticModel
from language_model import LanguageModel


VISUALIZATION_DIR = Path(__file__).resolve().parent / "visualization"


@dataclass
class Hypothesis:
    token_ids: list[int] = field(default_factory=list)
    text: str = ""
    last_frame: int = 0
    score: float = 0.0
    kv_cache: object = None
    finished: bool = False
    last_am_lp: float = 0.0
    last_lm_lp: float = 0.0

    def __repr__(self):
        # Don't show kv_cache in repr/printing
        fields = [
            f"token_ids={self.token_ids!r}",
            f"text={self.text!r}",
            f"last_frame={self.last_frame!r}",
            f"score={self.score!r}",
            f"finished={self.finished!r}",
            f"last_am_lp={self.last_am_lp!r}",
            f"last_lm_lp={self.last_lm_lp!r}",
        ]
        return f"Hypothesis({', '.join(fields)})"


class LLMGuidedDecoder:
    """Implements the beam-search loop of Algorithm 1."""

    def __init__(
        self,
        am: AcousticModel,
        lm: LanguageModel,
        cfg: Config | None = None,
        top_k_from_text: bool = False,
    ):
        self.am = am
        self.lm = lm
        self.cfg = cfg or Config()
        self.top_k_from_text = top_k_from_text
    # ------------------------------------------------------------------ #
    #  Stopping criteria                                                  #
    # ------------------------------------------------------------------ #
    def _is_finished(
        self,
        hyp: Hypothesis,
        emissions: torch.Tensor,
        eos_log_prob: float,
        best_cand_lm_lp: float,
    ) -> bool:
        """Check the three stopping conditions from Section VII."""
        T = emissions.shape[0]

        if len(hyp.token_ids) == 0:
            return False

        # (iii) Audio exhaustion
        if hyp.last_frame >= T:
            return True

        # (i) Completion: EOS more likely than extending
        if eos_log_prob > best_cand_lm_lp:
            return True

        return False

    def _check_acoustic_threshold(
        self, token_text: str, am_log_prob: float, n_chars: int
    ) -> bool:
        """Return True if this candidate should be KEPT (passes threshold).

        Condition (ii): terminate paths where acoustic prob for the token
        falls below threshold for non-whitespace tokens.
        Uses exp(am_log_prob) for the full token alignment.
        """
        if token_text.strip() == "":
            return True
        if n_chars == 0:
            return False
        prob = math.exp(am_log_prob)
        return prob >= self.cfg.acoustic_threshold

    # ------------------------------------------------------------------ #
    #  Main decode loop                                                   #
    # ------------------------------------------------------------------ #
    def decode(
        self,
        waveform: torch.Tensor,
        beam_width: int | None = None,
        top_k: int | None = None,
        alpha: float | None = None,
        beta: float | None = None,
        max_steps: int | None = None,
        max_lookahead: int | None = None,
        verbose: bool = False,
    ) -> Hypothesis:
        """Run Algorithm 1 and return the best hypothesis."""
        cfg = self.cfg
        B = beam_width or cfg.beam_width
        K = top_k or cfg.top_k
        alpha = alpha if alpha is not None else cfg.alpha
        beta = beta if beta is not None else cfg.beta
        max_steps = max_steps or cfg.max_steps
        max_la = max_lookahead or cfg.max_lookahead_frames

        emissions = self.am.get_emissions(waveform)  # (T, C)
        T = emissions.shape[0]
        print(f"Emissions shape: {emissions.shape}")

        # Save emission matrix visualization (similar to clasic_asr.py)
        if verbose:
            try:
                VISUALIZATION_DIR.mkdir(exist_ok=True)
                emissions_np = emissions.numpy()
                probs = np.exp(emissions_np)  # log-softmax -> prob
                n_frames, n_tokens = probs.shape
                sorted_indices = sorted(
                    range(n_tokens), key=lambda i: self.am.idx_to_char.get(i, "?")
                )
                probs_ordered = probs[:, sorted_indices]
                vocab_labels = [self.am.idx_to_char.get(i, "?") for i in sorted_indices]

                fig, ax = plt.subplots(
                    figsize=(14, max(6, n_tokens * 0.25))
                )
                im = ax.imshow(
                    probs_ordered.T, aspect="auto", origin="lower", cmap="viridis"
                )
                ax.set_xlabel("Time (frame)")
                ax.set_ylabel("Token")
                ax.set_yticks(range(n_tokens))
                ax.set_yticklabels(vocab_labels)
                ax.set_title("Emission matrix (P(token | frame))")
                ax.set_xlim(0, 25)
                plt.colorbar(im, ax=ax, label="Probability")
                plt.tight_layout()
                out_path = VISUALIZATION_DIR / "emission_matrix.png"
                plt.savefig(out_path, dpi=150)
                plt.close()
                print(f"Emission matrix saved to {out_path}")
            except Exception as e:
                print(f"Warning: failed to save emission matrix: {e}")

        beam: list[Hypothesis] = [Hypothesis()]  # line 2 in Algorithm 1
        beam_log_rows: list[dict] = []

        for step in range(1, max_steps + 1):
            candidates: list[Hypothesis] = []  # line 4 in Algorithm 1

            # Group hypotheses by prefix to avoid redundant LLM calls
            # TODO: figure out later 
            prefix_groups: dict[tuple, list[Hypothesis]] = {}
            for hyp in beam:
                if hyp.finished:
                    candidates.append(hyp)
                    continue
                key = tuple(hyp.token_ids)
                prefix_groups.setdefault(key, []).append(hyp)

            for key, hyps in prefix_groups.items():
                rep = hyps[0]  # representative hypothesis
                
                if self.top_k_from_text:
                    tok_ids, lm_lps = self.lm.top_k_from_text(rep.text, K)
                    new_kv = None
                else:
                    tok_ids, lm_lps, new_kv = self.lm.top_k(rep.token_ids, K, None) # rep.kv_cache)

                # Check EOS probability for stopping criterion (i)
                eos_lp = float("-inf")
                eos_idx = self.lm.eos_token_id
                if eos_idx is not None:
                    eos_mask = tok_ids == eos_idx
                    if eos_mask.any():
                        eos_lp = lm_lps[eos_mask][0].item()

                best_cand_lm_lp = lm_lps[0].item() if len(lm_lps) > 0 else float("-inf")

                for hyp in hyps:
                    if self._is_finished(hyp, emissions, eos_lp, best_cand_lm_lp): #TODO: understand this
                        hyp.finished = True
                        candidates.append(hyp)
                        continue

                    for tid, lm_lp in zip(tok_ids.tolist(), lm_lps.tolist()):
                        if tid == self.lm.eos_token_id:
                            if len(hyp.token_ids) == 0:
                                continue
                            fin = Hypothesis(
                                token_ids=hyp.token_ids[:],
                                text=hyp.text,
                                last_frame=hyp.last_frame,
                                score=hyp.score + alpha * lm_lp,
                                kv_cache=None,
                                finished=True,
                                last_am_lp=hyp.last_am_lp,
                                last_lm_lp=lm_lp,
                            )
                            candidates.append(fin)
                            # if verbose: print(f"fin: {fin}")
                            continue

                        token_text = self.lm.decode_token(tid)
                        end_frame, am_lp = self.am.align_token(
                            token_text, hyp.last_frame, emissions, max_la
                        ) # line 8 in Algorithm 1

                        char_count = len(
                            self.am.token_to_char_indices(token_text)
                        )
                        if not self._check_acoustic_threshold(
                            token_text, am_lp, char_count
                        ):
                            continue
                        
                        new_score = hyp.score + am_lp + alpha * lm_lp + beta # line 9 in Algorithm 1
                        new_hyp = Hypothesis(
                            token_ids=hyp.token_ids + [tid],
                            text=hyp.text + token_text,
                            last_frame=end_frame+1,
                            score=new_score,
                            kv_cache=new_kv,
                            finished=False,
                            last_am_lp=am_lp,
                            last_lm_lp=lm_lp,
                        )
                        candidates.append(new_hyp) # line 10 in Algorithm 1

            if not candidates:
                break

            # Keep top-B
            candidates.sort(key=lambda h: h.score, reverse=True)
            beam = candidates[:B]

            # Log beam state for visualization (excluding kv_cache)
            for rank, h in enumerate(beam):
                beam_log_rows.append(
                    {
                        "step": step,
                        "rank": rank,
                        "token_ids": " ".join(map(str, h.token_ids)),
                        "text": h.text,
                        "last_frame": h.last_frame,
                        "score": h.score,
                        "finished": h.finished,
                        "last_am_lp": h.last_am_lp,
                        "last_lm_lp": h.last_lm_lp,
                    }
                )

            # print the beam
            if verbose:
                print(f"beam: {beam}")

            best = beam[0]
            print(f"  step {step:3d}  score={best.score:8.2f}  "
                    f"frame={best.last_frame:4d}/{T}  "
                    f"text={best.text!r}")

            if all(h.finished for h in beam):
                print("All hypotheses finished. Exiting loop.")
                break

        # Save beam evolution to CSV (one file per decode) without kv_cache
        if beam_log_rows and verbose:
            try:
                os.makedirs("beam_logs", exist_ok=True)
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                rand_suffix = random.randint(0, 1_000_000)
                csv_path = os.path.join(
                    "beam_logs", f"beam_{timestamp}_{rand_suffix}.csv"
                )
                fieldnames = [
                    "step",
                    "rank",
                    "token_ids",
                    "text",
                    "last_frame",
                    "score",
                    "finished",
                    "last_am_lp",
                    "last_lm_lp",
                ]
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(beam_log_rows)
            except Exception as e:
                print(f"Warning: failed to write beam log CSV: {e}")

        beam.sort(key=lambda h: h.score, reverse=True)
        return beam[0]


# ------------------------------------------------------------------
# Test
# ------------------------------------------------------------------
def test():
    from test_utils import load_test_sample
    from preprocessing import preprocess
    from text_utils import normalize_for_eval

    print("=== decoder test ===\n")

    waveform, sr, ref = load_test_sample(chapter="121123", idx=0)
    processed, sr = preprocess(waveform=waveform, sr=sr, use_vad=False)
    ref_norm = normalize_for_eval(ref)
    print(f"Reference (norm): {ref_norm!r}")
    print(f"Audio duration  : {processed.shape[-1]/sr:.2f}s\n")

    am = AcousticModel()
    lm = LanguageModel()
    decoder = LLMGuidedDecoder(am, lm)

    # Small params for quick test
    best = decoder.decode(
        processed,
        beam_width=2,
        top_k=50,
        max_steps=15,
        verbose=True,
    )

    hyp_norm = normalize_for_eval(best.text)
    print(f"\nDecoded (norm) : {hyp_norm!r}")
    print(f"Reference      : {ref_norm!r}")
    print(f"Score          : {best.score:.2f}")
    print(f"Tokens         : {len(best.token_ids)}")
    print(f"Last frame     : {best.last_frame}")
    print(f"Finished       : {best.finished}")

    # Quick WER
    from jiwer import wer, cer
    w = wer(ref_norm, hyp_norm)
    c = cer(ref_norm, hyp_norm)
    print(f"WER            : {w:.2%}")
    print(f"CER            : {c:.2%}")
    print("\nDone.")


if __name__ == "__main__":
    test()
