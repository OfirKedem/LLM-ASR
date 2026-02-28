"""Acoustic model: wav2vec 2.0 CTC emissions + Viterbi forced alignment."""

import torch
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from config import Config

try:
    import numba
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def _viterbi_ctc(em: np.ndarray, labels: np.ndarray) -> tuple[float, int]:
    """CTC Viterbi DP: best log-prob and best end frame index.

    em: (T_local, C) float64 emission log-probs
    labels: (S,) int32 CTC expanded labels (blank, c1, blank, c2, ..., blank)
    Returns: (best_log_prob, best_t) where best_t is 0-based frame index
    """
    if _HAS_NUMBA:
        return _viterbi_ctc_numba(em, labels)
    return _viterbi_ctc_python(em, labels)


if _HAS_NUMBA:

    @numba.jit(nopython=True, cache=True)
    def _viterbi_ctc_numba(em: np.ndarray, labels: np.ndarray) -> tuple[float, int]:
        T_local, C = em.shape
        S = len(labels)
        NEG_INF = -1e9

        dp = np.full((T_local, S), NEG_INF, dtype=np.float64)
        dp[0, 0] = em[0, labels[0]]
        if S > 1:
            dp[0, 1] = em[0, labels[1]]

        for t in range(1, T_local):
            for s in range(S):
                best = dp[t - 1, s]
                if s >= 1:
                    best = max(best, dp[t - 1, s - 1])
                if s >= 2 and labels[s] != labels[s - 2]:
                    best = max(best, dp[t - 1, s - 2])
                dp[t, s] = best + em[t, labels[s]]

        U = (S - 1) // 2
        min_frames = U
        best_log_prob = NEG_INF
        best_t = T_local - 1
        for t in range(min_frames - 1, T_local):
            cand = max(dp[t, S - 1], dp[t, S - 2])
            if cand > best_log_prob:
                best_log_prob = cand
                best_t = t
        return float(best_log_prob), int(best_t)


def _viterbi_ctc_python(em: np.ndarray, labels: np.ndarray) -> tuple[float, int]:
    """Pure Python fallback when Numba is not available."""
    T_local, C = em.shape
    S = len(labels)
    NEG_INF = -1e9

    dp = np.full((T_local, S), NEG_INF, dtype=np.float64)
    dp[0, 0] = em[0, labels[0]]
    if S > 1:
        dp[0, 1] = em[0, labels[1]]

    for t in range(1, T_local):
        for s in range(S):
            best = dp[t - 1, s]
            if s >= 1:
                best = max(best, dp[t - 1, s - 1])
            if s >= 2 and labels[s] != labels[s - 2]:
                best = max(best, dp[t - 1, s - 2])
            dp[t, s] = best + em[t, labels[s]]

    U = (S - 1) // 2
    min_frames = U
    best_log_prob = NEG_INF
    best_t = T_local - 1
    for t in range(min_frames - 1, T_local):
        cand = max(dp[t, S - 1], dp[t, S - 2])
        if cand > best_log_prob:
            best_log_prob = cand
            best_t = t
    return float(best_log_prob), int(best_t)


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
        labels = np.empty(2 * U + 1, dtype=np.int32)
        for i, c in enumerate(char_ids):
            labels[2 * i] = self.blank_idx
            labels[2 * i + 1] = c
        labels[-1] = self.blank_idx

        em = emissions[start_frame:end_limit].numpy().astype(np.float64)
        best_log_prob, best_t = _viterbi_ctc(em, labels)
        end_frame = start_frame + best_t + 1
        
        if self.normalize_tokens is not None:  # mitigates bias towards shorter tokens
            best_log_prob = best_log_prob / (U * self.normalize_tokens)

        return end_frame, float(best_log_prob)

    def align_tokens_batch(
        self,
        token_texts: list[str],
        start_frame: int,
        emissions: torch.Tensor,
        max_lookahead: int = 75,
    ) -> list[tuple[int, float]]:
        """CTC Viterbi alignment for multiple tokens at once (Numba-accelerated).

        Returns list of (end_frame, log_prob) for each token_text.
        Empty/invalid tokens get (start_frame, 0.0).
        """
        if not token_texts:
            return []

        T_total = emissions.shape[0]
        end_limit = min(start_frame + max_lookahead, T_total)
        T_local = end_limit - start_frame
        if T_local < 1:
            return [(start_frame, float("-inf"))] * len(token_texts)

        em = emissions[start_frame:end_limit].numpy().astype(np.float64)
        results: list[tuple[int, float]] = []

        if _HAS_NUMBA:
            labels_list = []
            valid_indices: list[int] = []
            for token_text in token_texts:
                char_ids = self.token_to_char_indices(token_text)
                if not char_ids:
                    results.append((start_frame, 0.0))
                    continue
                U = len(char_ids)
                labels = np.empty(2 * U + 1, dtype=np.int32)
                for i, c in enumerate(char_ids):
                    labels[2 * i] = self.blank_idx
                    labels[2 * i + 1] = c
                labels[-1] = self.blank_idx
                labels_list.append(labels)
                valid_indices.append(len(results))
                results.append((0, 0.0))  # placeholder

            if labels_list:
                from numba.typed import List as NumbaList

                labels_numba = NumbaList()
                for lbl in labels_list:
                    labels_numba.append(lbl)

                best_lps, best_ts = _viterbi_ctc_batch_numba(em, labels_numba)
                for i in range(len(labels_list)):
                    idx = valid_indices[i]
                    lp = float(best_lps[i])
                    bt = int(best_ts[i])
                    end_frame = start_frame + bt + 1
                    U = (len(labels_list[i]) - 1) // 2
                    if self.normalize_tokens is not None:
                        lp = lp / (U * self.normalize_tokens)
                    results[idx] = (end_frame, lp)
        else:
            for token_text in token_texts:
                ef, lp = self.align_token(
                    token_text, start_frame, emissions, max_lookahead
                )
                results.append((ef, lp))

        return results


if _HAS_NUMBA:

    @numba.jit(nopython=True, parallel=True, cache=True)
    def _viterbi_ctc_batch_numba(em: np.ndarray, labels_list) -> tuple[np.ndarray, np.ndarray]:
        """Run Viterbi for multiple label sequences in parallel."""
        n = len(labels_list)
        best_log_probs = np.empty(n, dtype=np.float64)
        best_ts = np.empty(n, dtype=np.int64)
        for i in numba.prange(n):
            labels = labels_list[i]
            best_lp, best_t = _viterbi_ctc_numba(em, labels)
            best_log_probs[i] = best_lp
            best_ts[i] = best_t
        return best_log_probs, best_ts


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
