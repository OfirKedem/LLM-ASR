"""Main entry point: run the LLM-guided ASR decoder on audio files."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from config import Config
from acoustic_model import AcousticModel
from language_model import LanguageModel
from decoder import LLMGuidedDecoder
from preprocessing import preprocess
from evaluate import compute_wer, compute_cer, evaluate_batch
from text_utils import normalize_for_eval


def decode_file(
    audio_path: str,
    decoder: LLMGuidedDecoder,
    cfg: Config,
    verbose: bool = False,
) -> str:
    """Preprocess and decode a single audio file."""
    processed, sr = preprocess(path=audio_path, cfg=cfg)
    hyp = decoder.decode(processed, verbose=verbose)
    return hyp.text


def run_on_directory(
    data_dir: Path,
    decoder: LLMGuidedDecoder,
    cfg: Config,
    max_samples: int | None = None,
    verbose: bool = False,
):
    """Run decoder on all .flac files under *data_dir* that have transcripts."""
    from test_utils import _parse_trans

    results = []
    for trans_file in sorted(data_dir.rglob("*.trans.txt")):
        mapping = _parse_trans(trans_file)
        chapter_dir = trans_file.parent
        items = sorted(mapping.items())
        if max_samples is not None:
            items = items[:max_samples]
        for utt_id, ref_text in items:
            flac = chapter_dir / f"{utt_id}.flac"
            if not flac.exists():
                continue
            t0 = time.time()
            hyp_text = decode_file(str(flac), decoder, cfg, verbose=verbose)
            elapsed = time.time() - t0
            ref_n = normalize_for_eval(ref_text)
            hyp_n = normalize_for_eval(hyp_text)
            w = compute_wer(ref_text, hyp_text)
            c = compute_cer(ref_text, hyp_text)
            results.append({
                "utt_id": utt_id,
                "ref": ref_n,
                "hyp": hyp_n,
                "wer": w,
                "cer": c,
                "time": elapsed,
            })
            print(f"[{utt_id}]  WER={w:.2%}  CER={c:.2%}  ({elapsed:.1f}s)")
            print(f"  REF: {ref_n}")
            print(f"  HYP: {hyp_n}\n")

    if results:
        refs = [r["ref"] for r in results]
        hyps = [r["hyp"] for r in results]
        agg = evaluate_batch(refs, hyps, normalize=False)
        total_time = sum(r["time"] for r in results)
        print("=" * 60)
        print(f"Aggregate ({len(results)} utts): "
              f"WER={agg['wer']:.2%}  CER={agg['cer']:.2%}  "
              f"Total time={total_time:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="LLM-Guided ASR Decoder")
    parser.add_argument("input", help="Audio file (.flac/.wav) or data directory")
    parser.add_argument("--am", default=None, help="Acoustic model name")
    parser.add_argument("--lm", default=None, help="Language model name")
    parser.add_argument("--beam", type=int, default=None, help="Beam width")
    parser.add_argument("--topk", type=int, default=None, help="Top-K candidates")
    parser.add_argument("--alpha", type=float, default=None, help="LM weight")
    parser.add_argument("--beta", type=float, default=None, help="Token bonus")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max utterances per chapter (for directory mode)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    if args.am:
        cfg.am_model_name = args.am
    if args.lm:
        cfg.lm_model_name = args.lm
    if args.beam:
        cfg.beam_width = args.beam
    if args.topk:
        cfg.top_k = args.topk
    if args.alpha is not None:
        cfg.alpha = args.alpha
    if args.beta is not None:
        cfg.beta = args.beta

    am = AcousticModel(cfg.am_model_name, cfg.device)
    lm = LanguageModel(cfg.lm_model_name, cfg.device)
    dec = LLMGuidedDecoder(am, lm, cfg)

    inp = Path(args.input)
    if inp.is_file():
        hyp = decode_file(str(inp), dec, cfg, verbose=args.verbose)
        print(f"Transcription: {normalize_for_eval(hyp)}")
    elif inp.is_dir():
        run_on_directory(inp, dec, cfg, max_samples=args.max_samples,
                         verbose=args.verbose)
    else:
        print(f"Error: {inp} is not a file or directory")


# ------------------------------------------------------------------
# Test
# ------------------------------------------------------------------
def test():
    from test_utils import load_test_sample, load_chapter_samples

    print("=== run.py test ===\n")

    cfg = Config()
    cfg.beam_width = 5
    cfg.top_k = 5000
    cfg.max_steps = 15
    cfg.alpha = 0.0626
    cfg.beta = 0.5190

    am = AcousticModel(cfg.am_model_name, cfg.device)
    lm = LanguageModel(cfg.lm_model_name, cfg.device)
    dec = LLMGuidedDecoder(am, lm, cfg)

    # 1. Single file
    waveform, sr, ref = load_test_sample(chapter="121123", idx=27)
    processed, sr = preprocess(waveform=waveform, sr=sr, cfg=cfg, use_vad=False)
    print(f"Audio duration: {processed.shape[-1]/sr:.2f}s")
    hyp = dec.decode(processed, verbose=True)
    ref_n = normalize_for_eval(ref)
    hyp_n = normalize_for_eval(hyp.text)
    w = compute_wer(ref, hyp.text)
    c = compute_cer(ref, hyp.text)
    print(f"\nREF: {ref_n}")
    print(f"HYP: {hyp_n}")
    print(f"WER: {w:.2%}  CER: {c:.2%}\n")

    # # 2. Multiple utterances
    # print("--- Batch (3 utterances) ---")
    # samples = load_chapter_samples(chapter="121123", max_samples=3)
    # refs, hyps_text = [], []
    # for i, (wav, sr_i, ref_i) in enumerate(samples):
    #     proc, _ = preprocess(waveform=wav, sr=sr_i, cfg=cfg, use_vad=False)
    #     h = dec.decode(proc)
    #     refs.append(ref_i)
    #     hyps_text.append(h.text)
    #     print(f"  [{i}] REF: {normalize_for_eval(ref_i)}")
    #     print(f"      HYP: {normalize_for_eval(h.text)}")
    #     print(f"      WER: {compute_wer(ref_i, h.text):.2%}")

    # agg = evaluate_batch(refs, hyps_text)
    # print(f"\nAggregate: WER={agg['wer']:.2%}  CER={agg['cer']:.2%}")
    print("\nDone.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] != "--test":
        main()
    else:
        test()
