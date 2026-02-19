"""Evaluation utilities: WER and CER computation."""

from jiwer import wer as _jiwer_wer, cer as _jiwer_cer

from text_utils import normalize_for_eval


def compute_wer(reference: str, hypothesis: str, normalize: bool = True) -> float:
    ref = normalize_for_eval(reference) if normalize else reference
    hyp = normalize_for_eval(hypothesis) if normalize else hypothesis
    if not ref:
        return 0.0 if not hyp else 1.0
    return _jiwer_wer(ref, hyp)


def compute_cer(reference: str, hypothesis: str, normalize: bool = True) -> float:
    ref = normalize_for_eval(reference) if normalize else reference
    hyp = normalize_for_eval(hypothesis) if normalize else hypothesis
    if not ref:
        return 0.0 if not hyp else 1.0
    return _jiwer_cer(ref, hyp)


def evaluate_batch(
    references: list[str],
    hypotheses: list[str],
    normalize: bool = True,
) -> dict[str, float]:
    """Compute corpus-level WER and CER over parallel lists."""
    refs = [normalize_for_eval(r) if normalize else r for r in references]
    hyps = [normalize_for_eval(h) if normalize else h for h in hypotheses]
    return {
        "wer": _jiwer_wer(refs, hyps),
        "cer": _jiwer_cer(refs, hyps),
    }


# ------------------------------------------------------------------
# Test
# ------------------------------------------------------------------
def test():
    print("=== evaluate test ===\n")

    # 1. Perfect match
    w = compute_wer("go do you hear", "go do you hear")
    c = compute_cer("go do you hear", "go do you hear")
    print(f"Perfect match:  WER={w:.4f}  CER={c:.4f}  (expect 0.0)")

    # 2. One-word error
    w2 = compute_wer("go do you hear", "go do you here")
    c2 = compute_cer("go do you hear", "go do you here")
    print(f"One word diff:  WER={w2:.4f}  CER={c2:.4f}  (expect WER=0.25)")

    # 3. Completely wrong
    w3 = compute_wer("go do you hear", "the cat sat down")
    c3 = compute_cer("go do you hear", "the cat sat down")
    print(f"All wrong:      WER={w3:.4f}  CER={c3:.4f}")

    # 4. Normalization handles uppercase + punctuation
    w4 = compute_wer("GO DO YOU HEAR!", "go do you hear")
    print(f"Normalize test: WER={w4:.4f}  (expect 0.0)")

    # 5. Batch from real data using greedy CTC
    print("\nGreedy CTC on 5 real utterances:")
    from test_utils import load_chapter_samples
    from preprocessing import preprocess
    from acoustic_model import AcousticModel

    am = AcousticModel()
    samples = load_chapter_samples(chapter="121123", max_samples=5)
    refs, hyps = [], []
    for waveform, sr, ref_text in samples:
        processed, sr = preprocess(waveform=waveform, sr=sr, use_vad=False)
        emissions = am.get_emissions(processed)
        greedy = am.greedy_decode(emissions)
        refs.append(ref_text)
        hyps.append(greedy)
        r_n = normalize_for_eval(ref_text)
        h_n = normalize_for_eval(greedy)
        w = compute_wer(ref_text, greedy)
        print(f"  REF: {r_n}")
        print(f"  HYP: {h_n}")
        print(f"  WER: {w:.2%}\n")

    batch = evaluate_batch(refs, hyps)
    print(f"Aggregate:  WER={batch['wer']:.2%}  CER={batch['cer']:.2%}")
    print("\nDone.")


if __name__ == "__main__":
    test()
