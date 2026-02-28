# LLM-Guided ASR Decoder

Zero-shot ASR decoder that uses a pretrained acoustic model (wav2vec2) and a language model (GPT-2) for beam-search decoding. Based on *LLM Guided Decoding for SSL-ASR* (arXiv:2508.02228v2).

## Setup

Use the conda environment named **speech** and install dependencies:

```bash
conda activate speech
pip install -r requirements.txt
```

Requirements include: PyTorch, transformers, wav2vec2, GPT-2, datasets, numba, soundfile, jiwer, wandb (optional).

## How to Run

### Single audio file

Decode one `.flac` or `.wav` file:

```bash
python run.py path/to/audio.flac
```

With options (override defaults from `config.py`):

```bash
python run.py path/to/audio.flac --beam 5 --topk 5000 --alpha 0.3 --beta 2.0 --verbose
```

### Directory (LibriSpeech-style)

Run on all utterances under a directory that has `.trans.txt` and matching `.flac` files (e.g. a LibriSpeech chapter):

```bash
python run.py data/84/121123/ --max-samples 5
```

Limiting with `--max-samples` is useful for quick runs. Use `--wandb` to log WER/CER to Weights & Biases.

### Built-in test

Run the in-repo test (single sample from `data/84/121123`, no CLI input):

```bash
python run.py --test
```

### Benchmark datasets (LibriSpeech, TED-LIUM 3)

Evaluate on HuggingFace datasets with streaming:

```bash
python run_dataset.py librispeech_dummy --max-samples 10   # small dummy set
python run_dataset.py librispeech --max-samples 100         # LibriSpeech test-clean
python run_dataset.py tedlium --max-samples 50              # TED-LIUM 3 test
```

Use `--paper` for paper settings (beam=5, topk=5000) and `--use-vad` for VAD preprocessing.

### Hyperparameter sweep (W&B)

Minimize WER over `beam_width`, `alpha`, and `beta` on a fixed data path:

1. Create a sweep from `sweep.yaml` (edit the `command` input path if needed):
   ```bash
   wandb sweep sweep.yaml
   ```
2. Run an agent with the printed sweep ID:
   ```bash
   wandb agent <sweep_id>
   ```

Sweep runs use `--wandb` and log WER so Bayesian search can minimize it.

## Main options (`run.py`)

| Option        | Description                    | Default (from config) |
|---------------|--------------------------------|------------------------|
| `input`       | Audio file or data directory   | —                      |
| `--beam`      | Beam width                     | 5                      |
| `--topk`      | Top-K LM candidates            | 5000                   |
| `--alpha`     | LM weight                      | ~0.30                  |
| `--beta`      | Token insertion bonus          | ~1.99                  |
| `--max-steps` | Max decoding steps             | 1000                   |
| `--max-samples` | Max utterances (dir mode)    | None                   |
| `--verbose`   | Extra logs + beam CSV + emission viz | False        |
| `--wandb`     | Log to Weights & Biases        | False                  |

## Visualizing the beam

When you run with `--verbose`, the decoder writes a beam log CSV under `beam_logs/`. To animate it:

```bash
python visualize_beam.py beam_logs/beam_YYYYMMDD-HHMMSS_*.csv --mode gif --output beam_viz.gif
```

Modes: `terminal` (animate in terminal), `gif`, or `mp4` (if ffmpeg available).

## Data

### Local data folder layout (LibriSpeech-style)

```
data/
├── 84/                          # speaker/folder id
│   ├── 121123/                  # chapter id
│   │   ├── 84-121123.trans.txt  # "utt_id TRANSCRIPT" per line
│   │   ├── 84-121123-0000.flac
│   │   ├── 84-121123-0001.flac
│   │   └── ...
│   └── 121550/
│       ├── 84-121550.trans.txt
│       └── *.flac
└── 61/
    ├── 70968/
    │   ├── 61-70968.trans.txt
    │   └── *.flac
    └── 70970/
        ├── 61-70970.trans.txt
        └── *.flac
```

- **Local LibriSpeech-style**: Put `.flac` files and a `*.trans.txt` (utt_id + transcript per line) in the same directory; point `run.py` at that directory (e.g. `data/84/121123/`).
- **Benchmarks**: `run_dataset.py` pulls LibriSpeech and TED-LIUM 3 from HuggingFace with streaming.

## Config

Defaults live in `config.py`: acoustic model (`facebook/wav2vec2-base-960h`), language model (`gpt2`), beam width, top-k, alpha, beta, max_steps, device (`cuda`), etc. Override via CLI when calling `run.py` or `run_dataset.py`.
