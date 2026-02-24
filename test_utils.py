"""Shared helpers for loading real test samples from data/84/ and data/61/."""

from pathlib import Path
import torch
import soundfile as sf


DATA_DIR = Path(__file__).resolve().parent / "data"

# Available data folders under DATA_DIR
DATA_FOLDERS = ("84", "61")

# Available chapters per folder: (folder, chapter_id) -> path
CHAPTERS = {
    ("84", "121123"): DATA_DIR / "84" / "121123",
    ("84", "121550"): DATA_DIR / "84" / "121550",
    ("61", "70968"): DATA_DIR / "61" / "70968",
    ("61", "70970"): DATA_DIR / "61" / "70970",
}


def _parse_trans(trans_path: Path) -> dict[str, str]:
    """Parse a LibriSpeech .trans.txt into {utt_id: text}."""
    mapping: dict[str, str] = {}
    with open(trans_path) as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                mapping[parts[0]] = parts[1]
    return mapping


def load_test_sample(
    chapter: str = "121123",
    idx: int = 0,
    folder: str = "84",
) -> tuple[torch.Tensor, int, str]:
    """Return (waveform, sample_rate, reference_text) for one utterance.

    Parameters
    ----------
    chapter : str
        Chapter id – "121123" (short sentences) or "121550" (longer).
    idx : int
        Utterance index within the chapter (0-based).
    folder : str
        Data folder under DATA_DIR – "84" or "61".
    """
    chapter_dir = CHAPTERS[(folder, chapter)]
    trans_path = chapter_dir / f"{folder}-{chapter}.trans.txt"
    mapping = _parse_trans(trans_path)
    utt_ids = sorted(mapping.keys())
    utt_id = utt_ids[idx]
    ref_text = mapping[utt_id]
    flac_path = chapter_dir / f"{utt_id}.flac"
    data, sr = sf.read(str(flac_path), dtype="float32")
    waveform = torch.from_numpy(data).unsqueeze(0)  # (1, N)
    return waveform, sr, ref_text


def load_chapter_samples(
    chapter: str = "121123",
    max_samples: int | None = None,
    folder: str = "84",
) -> list[tuple[torch.Tensor, int, str]]:
    """Load all (or first *max_samples*) utterances from a chapter."""
    chapter_dir = CHAPTERS[(folder, chapter)]
    trans_path = chapter_dir / f"{folder}-{chapter}.trans.txt"
    mapping = _parse_trans(trans_path)
    utt_ids = sorted(mapping.keys())
    if max_samples is not None:
        utt_ids = utt_ids[:max_samples]
    samples = []
    for utt_id in utt_ids:
        flac_path = chapter_dir / f"{utt_id}.flac"
        data, sr = sf.read(str(flac_path), dtype="float32")
        waveform = torch.from_numpy(data).unsqueeze(0)
        samples.append((waveform, sr, mapping[utt_id]))
    return samples
