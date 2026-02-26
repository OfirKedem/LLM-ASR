"""Audio preprocessing: loading, silence padding, Silero VAD trimming."""

import torch
import soundfile as sf
import numpy as np

from config import Config


def load_audio(path: str, target_sr: int = 16_000) -> tuple[torch.Tensor, int]:
    """Load audio file and resample to *target_sr* Hz mono."""
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    waveform = torch.from_numpy(data).unsqueeze(0)  # (1, N)
    if sr != target_sr:
        import torchaudio
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform, target_sr


def pad_silence(
    waveform: torch.Tensor,
    sr: int,
    pad_seconds: float = 0.5,
) -> torch.Tensor:
    """Prepend *pad_seconds* of silence to the waveform."""
    pad_samples = int(sr * pad_seconds)
    silence = torch.zeros(1, pad_samples, dtype=waveform.dtype)
    return torch.cat([silence, waveform], dim=-1)


_vad_cache = None


def _get_vad():
    """Load Silero VAD model (cached). Returns None if loading fails."""
    global _vad_cache
    if _vad_cache is not None:
        return _vad_cache
    try:
        vad_model, utils = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True
        )
        _vad_cache = (vad_model, utils)
        return _vad_cache
    except Exception:
        return None


def trim_with_vad(
    waveform: torch.Tensor,
    sr: int,
    start_extension: float = 0.2,
    end_extension: float = 0.0,
) -> torch.Tensor:
    """Use Silero VAD to remove leading/trailing silence. Falls back to no-op if VAD unavailable."""
    vad_result = _get_vad()
    if vad_result is None:
        return waveform

    vad_model, utils = vad_result
    get_speech_timestamps = utils[0]
    wav = waveform.squeeze()
    if wav.abs().max() > 1.0:
        wav = wav / wav.abs().max()

    timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=sr)
    if not timestamps:
        return waveform

    start_sample = timestamps[0]["start"]
    end_sample = timestamps[-1]["end"]

    start_sample = max(0, start_sample - int(start_extension * sr))
    end_sample = min(waveform.shape[-1], end_sample + int(end_extension * sr))

    return waveform[..., start_sample:end_sample]


def preprocess(
    path: str | None = None,
    waveform: torch.Tensor | None = None,
    sr: int | None = None,
    cfg: Config | None = None,
    use_vad: bool = True,
) -> tuple[torch.Tensor, int]:
    """Full preprocessing pipeline.

    Either supply *path* (audio file) or (*waveform*, *sr*).
    Returns (processed_waveform, sample_rate).
    If use_vad is False, VAD trimming is skipped (useful when Silero is unavailable).
    """
    if cfg is None:
        cfg = Config()

    if path is not None:
        waveform, sr = load_audio(path, cfg.sample_rate)
    assert waveform is not None and sr is not None

    waveform = pad_silence(waveform, sr, cfg.silence_pad_seconds)
    if use_vad:
        waveform = trim_with_vad(
            waveform, sr, cfg.vad_start_extension, cfg.vad_end_extension
        )
    return waveform, sr


# ------------------------------------------------------------------
# Test
# ------------------------------------------------------------------
def test():
    from test_utils import load_test_sample

    print("=== preprocessing test ===\n")

    waveform, sr, ref = load_test_sample(folder="84", chapter="121123", idx=0)
    print(f"Reference text : {ref}")
    print(f"Sample rate    : {sr}")
    print(f"Original shape : {waveform.shape}  ({waveform.shape[-1]/sr:.2f}s)")

    padded = pad_silence(waveform, sr, 0.5)
    print(f"After padding  : {padded.shape}  ({padded.shape[-1]/sr:.2f}s)")

    processed, out_sr = preprocess(waveform=waveform, sr=sr, use_vad=True)
    print(f"Full pipeline  : {processed.shape}  ({processed.shape[-1]/out_sr:.2f}s)  [use_vad=True for test]")
    print(f"Waveform stats : min={processed.min():.4f}  max={processed.max():.4f}  "
          f"mean={processed.mean():.4f}")

    assert processed.dim() == 2
    assert processed.shape[0] == 1
    assert processed.shape[-1] > 0
    print("\nAll assertions passed.")

    # save the processed waveform to a file
    output_path = "processed_audio.wav"
    # processed is expected to be [channel, samples], convert if necessary
    # soundfile expects channels x samples or samples x channels for write
    # Here, processed.shape should be (1, n), so squeeze the channel dimension
    sf.write(output_path, processed.squeeze(), out_sr)
    print(f"Processed waveform saved to {output_path}")
    


if __name__ == "__main__":
    test()
