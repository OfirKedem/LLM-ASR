"""
Evaluate the LLM-guided ASR decoder on standard datasets from the paper:
- LibriSpeech test-clean
- TED-LIUM 3 (release3) test set

Paper reference: arXiv:2508.02228v2 (LLM Guided Decoding for SSL-ASR)
Reported results (wav2vec2-base-960h + GPT-2):
  - LibriSpeech: not in paper; included as common benchmark
  - TED-LIUM 3: WER 11.15%, CER 4.87%
"""

from __future__ import annotations

import argparse
import io
import time
import torch

from config import Config
from acoustic_model import AcousticModel
from language_model import LanguageModel
from decoder import LLMGuidedDecoder
from preprocessing import preprocess, load_audio
from evaluate import compute_wer, compute_cer, evaluate_batch
from text_utils import normalize_for_eval


DATASETS = {
    "librispeech": {
        "hf_path": "openslr/librispeech_asr",
        "config": "clean",
        "split": "test",
        "text_key": "text",
    },
    "librispeech_dummy": {
        "hf_path": "hf-internal-testing/librispeech_asr_dummy",
        "config": None,
        "split": "validation",
        "text_key": "text",
    },
    "tedlium": {
        "hf_path": "LIUM/tedlium",
        "config": "release3",
        "split": "test",
        "text_key": "text",
    },
}


def load_dataset_samples(name: str, max_samples: int | None = None):
    """Load (audio_tensor, sr, ref_text) tuples from a HuggingFace dataset."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Choose from: {list(DATASETS)}")

    cfg = DATASETS[name]
    print(f"Loading {name} from HuggingFace ({cfg['hf_path']}, {cfg['split']})...")
    load_kw = {"trust_remote_code": cfg.get("trust_remote_code", False)}
    load_args = [cfg["hf_path"]]
    if cfg.get("config"):
        load_args.append(cfg["config"])
    datasets = __import__("datasets")
    ds = datasets.load_dataset(
        *load_args,
        split=cfg["split"],
        **load_kw,
    )
    # Avoid torchcodec/FFmpeg by using decode=False and loading with soundfile
    ds = ds.cast_column(
        "audio",
        datasets.features.Audio(decode=False, sampling_rate=16000),
    )

    text_key = cfg["text_key"]
    samples = []
    import soundfile as sf

    for i, item in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break
        audio = item["audio"]
        if audio is None:
            continue
        text = item.get(text_key, item.get("text", ""))
        if not text:
            continue
        # Load from path or bytes (avoid datasets' torchcodec decode)
        path = audio.get("path") if isinstance(audio, dict) else None
        data, sr = None, 16000
        if path:
            try:
                data, sr = sf.read(path, dtype="float32")
            except Exception:
                path = None
        if path is None and isinstance(audio, dict) and audio.get("bytes"):
            data, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
        if data is None:
            continue
        if data.ndim > 1:
            data = data.mean(axis=1)
        waveform = torch.from_numpy(data).float().unsqueeze(0)
        samples.append((waveform, sr, text))

    print(f"Loaded {len(samples)} samples")
    return samples


def run_evaluation(
    dataset_name: str,
    max_samples: int | None = None,
    beam_width: int = 2,
    top_k: int = 100,
    use_vad: bool = False,
    verbose: bool = False,
):
    cfg = Config()
    cfg.beam_width = beam_width
    cfg.top_k = top_k

    samples = load_dataset_samples(dataset_name, max_samples=max_samples)
    if not samples:
        print("No samples loaded.")
        return

    am = AcousticModel(cfg.am_model_name, cfg.device)
    lm = LanguageModel(cfg.lm_model_name, cfg.device)
    dec = LLMGuidedDecoder(am, lm, cfg)

    refs, hyps = [], []
    total_time = 0.0

    for i, (waveform, sr, ref_text) in enumerate(samples):
        t0 = time.time()
        processed, _ = preprocess(waveform=waveform, sr=sr, cfg=cfg, use_vad=use_vad)
        hyp = dec.decode(processed, verbose=verbose)
        elapsed = time.time() - t0
        total_time += elapsed

        ref_n = normalize_for_eval(ref_text)
        hyp_n = normalize_for_eval(hyp.text)
        refs.append(ref_n)
        hyps.append(hyp_n)

        w = compute_wer(ref_text, hyp.text)
        print(f"[{i+1}/{len(samples)}]  WER={w:.2%}  ({elapsed:.1f}s)")
        if verbose or w > 0.5:
            print(f"  REF: {ref_n[:80]}{'...' if len(ref_n) > 80 else ''}")
            print(f"  HYP: {hyp_n[:80]}{'...' if len(hyp_n) > 80 else ''}")

    metrics = evaluate_batch(refs, hyps, normalize=False)
    print("\n" + "=" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Samples: {len(samples)}")
    print(f"WER: {metrics['wer']:.2%}")
    print(f"CER: {metrics['cer']:.2%}")
    print(f"Total time: {total_time:.1f}s  ({total_time/len(samples):.1f}s/utt)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate on paper datasets")
    parser.add_argument(
        "dataset",
        choices=list(DATASETS),
        help="Dataset name (librispeech or tedlium)",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples")
    parser.add_argument("--beam", type=int, default=2, help="Beam width")
    parser.add_argument("--topk", type=int, default=100, help="Top-K candidates")
    parser.add_argument("--paper", action="store_true", help="Use paper settings (beam=5, topk=5000) for reproduction")
    parser.add_argument("--use-vad", action="store_true", help="Use Silero VAD")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.paper:
        args.beam, args.topk = 5, 5000
        print("Using paper settings: beam=5, topk=5000")

    run_evaluation(
        args.dataset,
        max_samples=args.max_samples,
        beam_width=args.beam,
        top_k=args.topk,
        use_vad=args.use_vad,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
