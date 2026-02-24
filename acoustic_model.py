"""Acoustic model: wav2vec 2.0 CTC emissions + Viterbi forced alignment."""

import torch
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from config import Config


class AcousticModel:
    def __init__(self, 
        model_name: str | None = None, 
        device: str | None = None,
        sample_rate: int = 16_000,
        normalize_tokens: float = None):
        cfg = Config()
        model_name = model_name or cfg.am_model_name
        self.device = device or cfg.device

        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
        self.model.eval()

        vocab = self.processor.tokenizer.get_vocab()
        self.vocab = vocab
        self.idx_to_char = {v: k for k, v in vocab.items()}
        self.blank_idx = self.processor.tokenizer.pad_token_id  # CTC blank
        self.space_idx = vocab.get("|", vocab.get(" ", None))
        self.normalize_tokens = normalize_tokens

    # ------------------------------------------------------------------ #
    #  CTC emissions                                                      #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def get_emissions(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return CTC log-probabilities, shape ``(T, C)``."""
        wav = waveform.squeeze()
        inputs = self.processor(
            wav.cpu().numpy(), sampling_rate=16_000, return_tensors="pt", padding=True
        )
        input_values = inputs.input_values.to(self.device)
        logits = self.model(input_values).logits  # (1, T, C)
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs.squeeze(0).cpu()  # (T, C)

    # ------------------------------------------------------------------ #
    #  Greedy CTC decode (baseline / test helper)                         #
    # ------------------------------------------------------------------ #
    def greedy_decode(self, log_probs: torch.Tensor) -> str:
        """Argmax CTC decode: collapse repeats, remove blanks."""
        ids = log_probs.argmax(dim=-1).tolist()
        prev = None
        chars: list[str] = []
        for idx in ids:
            if idx == prev:
                continue
            prev = idx
            if idx == self.blank_idx:
                continue
            ch = self.idx_to_char.get(idx, "")
            if ch == "|":
                ch = " "
            chars.append(ch)
        return "".join(chars).strip()

    # ------------------------------------------------------------------ #
    #  Viterbi forced alignment of a single token                         #
    # ------------------------------------------------------------------ #
    def token_to_char_indices(self, token_text: str) -> list[int]:
        """Map a text string to wav2vec character indices.

        Spaces in *token_text* are mapped to the ``|`` (word-boundary) index.
        Characters not in the vocabulary are skipped.
        wav2vec2-base-960h uses uppercase letters in its vocab.
        """
        indices: list[int] = []
        for ch in token_text.lower():
            if ch == " ":
                if self.space_idx is not None:
                    indices.append(self.space_idx)
            else:
                ch_upper = ch.upper()
                if ch_upper in self.vocab:
                    indices.append(self.vocab[ch_upper])
        return indices

    def align_token(
        self,
        token_text: str,
        start_frame: int,
        emissions: torch.Tensor,
        max_lookahead: int = 75,
    ) -> tuple[int, float]:
        """CTC Viterbi forced alignment of *token_text* starting at *start_frame*.

        Returns
        -------
        end_frame : int
            Frame index one past the last frame consumed by this token.
        log_prob : float
            Total log-probability of the best alignment path.
        """
        char_ids = self.token_to_char_indices(token_text)
        if not char_ids:
            return start_frame, 0.0

        T_total = emissions.shape[0]
        end_limit = min(start_frame + max_lookahead, T_total)
        T_local = end_limit - start_frame
        U = len(char_ids)

        if T_local < 1:
            return start_frame, float("-inf")

        # Build CTC expanded label sequence: blank c1 blank c2 ... cU blank
        S = 2 * U + 1
        labels = []
        for c in char_ids:
            labels.append(self.blank_idx)
            labels.append(c)
        labels.append(self.blank_idx)

        NEG_INF = -1e9
        # dp[t, s] = best log-prob ending at local time t in state s
        dp = np.full((T_local, S), NEG_INF, dtype=np.float64)
        em = emissions[start_frame:end_limit].numpy().astype(np.float64)

        # Initialise: can start in state 0 (blank) or state 1 (first char)
        dp[0, 0] = em[0, labels[0]]
        if S > 1:
            dp[0, 1] = em[0, labels[1]]

        for t in range(1, T_local):
            for s in range(S):
                best = dp[t - 1, s]                          # stay
                if s >= 1:
                    best = max(best, dp[t - 1, s - 1])       # from prev state
                if s >= 2 and labels[s] != labels[s - 2]:
                    best = max(best, dp[t - 1, s - 2])       # skip blank
                dp[t, s] = best + em[t, labels[s]]

        # Best ending: must finish in last char (S-2) or trailing blank (S-1)
        best_log_prob = NEG_INF
        best_t = T_local - 1
        # Need at least U frames to represent U characters
        min_frames = U
        for t in range(min_frames - 1, T_local):
            cand = max(dp[t, S - 1], dp[t, S - 2])
            if cand > best_log_prob:
                best_log_prob = cand
                best_t = t

        end_frame = start_frame + best_t + 1
        
        if self.normalize_tokens is not None: # midigates bias towards shorter tokens
            best_log_prob = best_log_prob / (U**self.normalize_tokens)
        
        return end_frame, float(best_log_prob)


# ------------------------------------------------------------------
# Test
# ------------------------------------------------------------------
def test():
    from test_utils import load_test_sample
    from preprocessing import preprocess

    print("=== acoustic_model test ===\n")

    waveform, sr, ref = load_test_sample(chapter="121123", idx=0)
    print(f"Reference: {ref}")
    processed, sr = preprocess(waveform=waveform, sr=sr, use_vad=False)
    print(f"Audio: {processed.shape[-1]/sr:.2f}s\n")

    am = AcousticModel()

    # 1. Emissions
    emissions = am.get_emissions(processed)
    print(f"Emissions shape: {emissions.shape}  (T={emissions.shape[0]}, C={emissions.shape[1]})")
    print(f"Log-prob range : [{emissions.min():.2f}, {emissions.max():.2f}]")
    probs_sum = emissions.exp().sum(dim=-1)
    print(f"Prob sum check : min={probs_sum.min():.4f}  max={probs_sum.max():.4f}  "
          f"(should be ~1.0)\n")

    # 2. Greedy decode
    greedy = am.greedy_decode(emissions)
    print(f"Greedy decode  : {greedy!r}")
    print(f"Reference      : {ref!r}\n")

    # 3. Align first word: "go"
    end1_g, lp1_g = am.align_token("G", 0, emissions)
    print(f'align_token("G", start=0) -> end_frame={end1_g}, log_prob={lp1_g:.4f}')

    end1, lp1 = am.align_token("go", 0, emissions)
    print(f'align_token("go", start=0) -> end_frame={end1}, log_prob={lp1:.4f}')

    # 4. Align second word: "do"
    end2, lp2 = am.align_token(" do", end1, emissions)
    print(f'align_token(" do", start={end1}) -> end_frame={end2}, log_prob={lp2:.2f}')

    d_end2, d_lp2 = am.align_token(" d", end1, emissions)
    print(f'align_token(" d", start={end1}) -> end_frame={d_end2}, log_prob={d_lp2:.2f}')

    # 5. Align third word: "you"
    end3, lp3 = am.align_token(" you", end2, emissions)
    print(f'align_token(" you", start={end2}) -> end_frame={end3}, log_prob={lp3:.2f}')

    # 6. Align fourth word: "hear"
    end4, lp4 = am.align_token(" hear", end3, emissions)
    print(f'align_token(" hear", start={end3}) -> end_frame={end4}, log_prob={lp4:.2f}')

    total_frames = emissions.shape[0]
    print(f"\nTotal frames: {total_frames},  last alignment ended at frame {end4}")
    print("Done.")


if __name__ == "__main__":
    test()
