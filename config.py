from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # --- Model identifiers ---
    am_model_name: str = "facebook/wav2vec2-base-960h"
    lm_model_name: str = "gpt2"

    # --- Decoding hyper-parameters (Table I, GPT-2 / WSJ0) ---
    beam_width: int = 5
    top_k: int = 5000
    alpha: float = 0.29635080096811683      # LM weight
    beta: float = 1.9865548403964204       # token insertion bonus
    max_steps: int = 1000        # safety horizon (max decoding iterations)

    # --- Acoustic alignment ---
    max_lookahead_frames: int = 75   # 1500 ms forward boundary
    acoustic_threshold: float = 0.3  # min acoustic prob for token alignment (paper uses 0.3; 1e-9 avoids over-filtering with Viterbi path probs)

    # --- Preprocessing ---
    sample_rate: int = 16_000
    silence_pad_seconds: float = 0.5
    vad_start_extension: float = 0.2
    vad_end_extension: float = 0.0

    # --- Paths ---
    data_dir: Path = Path(__file__).resolve().parent / "data"

    # --- Device ---
    device: str = "cuda"
