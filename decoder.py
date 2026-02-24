"""Algorithm 1 – Zero-Shot LLM-Driven ASR Decoder."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import torch

from config import Config
from acoustic_model import AcousticModel
from language_model import LanguageModel


@dataclass
class Hypothesis:
    token_ids: list[int] = field(default_factory=list)
    text: str = ""
    last_frame: int = 0
    score: float = 0.0
    kv_cache: object = None
    finished: bool = False


class LLMGuidedDecoder:
    """Implements the beam-search loop of Algorithm 1."""

    def __init__(
        self,
        am: AcousticModel,
        lm: LanguageModel,
        cfg: Config | None = None,
    ):
        self.am = am
        self.lm = lm
        self.cfg = cfg or Config()

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

        beam: list[Hypothesis] = [Hypothesis()] # line 2 in Algorithm 1

        for step in range(1, max_steps + 1):
            candidates: list[Hypothesis] = []  # line 4 in Algorithm 1
            # print the beam
            print(f"beam: {beam}")

            # Group hypotheses by prefix to avoid redundant LLM calls
            # TODO: figure out later 
            prefix_groups: dict[tuple, list[Hypothesis]] = {}
            for hyp in beam:
                if hyp.finished:
                    # candidates.append(hyp)
                    continue
                key = tuple(hyp.token_ids)
                prefix_groups.setdefault(key, []).append(hyp)

            for key, hyps in prefix_groups.items():
                rep = hyps[0]  # representative hypothesis
                # tok_ids, lm_lps, new_kv = self.lm.top_k(
                #     rep.token_ids, K, rep.kv_cache
                # )
                tok_ids, lm_lps = self.lm.top_k_from_text(rep.text, K)
                new_kv = None

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
                            )
                            candidates.append(fin)
                            print(f"fin: {fin}, lm_lp: {lm_lp}")
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

                        if token_text in [" y", " you"] and hyp.text in ['Go dO', "go do"]:
                            print("--------------------------------")
                            print(f"token_text: {token_text}, am_lp: {am_lp}, lm_lp: {lm_lp}")
                            print(f"hyp.score: {hyp.score}, alpha: {alpha}, beta: {beta}")
                            print(f"new_score: {hyp.score + am_lp + alpha * lm_lp + beta}")
                            print(f"hyp.token_ids: {hyp.token_ids}")
                            print(f"hyp.text: {hyp.text}")
                            print(f"hyp.last_frame: {hyp.last_frame}")
                            print(f"hyp.score: {hyp.score}")
                            print(f"hyp.finished: {hyp.finished}")
                            print(f"tid: {tid}")
                        
                        new_score = hyp.score + am_lp + alpha * lm_lp + beta # line 9 in Algorithm 1
                        new_hyp = Hypothesis(
                            token_ids=hyp.token_ids + [tid],
                            text=hyp.text + token_text,
                            last_frame=end_frame,
                            score=new_score,
                            kv_cache=new_kv,
                            finished=False,
                        )
                        candidates.append(new_hyp) # line 10 in Algorithm 1

            if not candidates:
                break

            # Keep top-B
            candidates.sort(key=lambda h: h.score, reverse=True)
            beam = candidates[:B]

            if verbose:
                best = beam[0]
                print(f"  step {step:3d}  score={best.score:8.2f}  "
                      f"frame={best.last_frame:4d}/{T}  "
                      f"text={best.text!r}")

            if all(h.finished for h in beam):
                break

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
