from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchaudio.models.decoder import ctc_decoder
from torchaudio.models.decoder import download_pretrained_files

from acoustic_model import AcousticModel
from test_utils import load_test_sample
from preprocessing import preprocess

VISUALIZATION_DIR = Path(__file__).resolve().parent / "visualization"


# 1. Get the emissions from your existing class

print("=== acoustic_model test ===\n")

waveform, sr, ref = load_test_sample(folder="61", chapter="70968", idx=2)
print(f"Reference: {ref}")

# Optional: add Gaussian white noise (set ADD_NOISE=False to disable)
ADD_NOISE = False
NOISE_SNR_DB = 1.0  # signal-to-noise ratio in dB (higher = cleaner)
if ADD_NOISE:
    signal_power = waveform.pow(2).mean()
    snr_linear = 10 ** (NOISE_SNR_DB / 10.0)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(waveform) * (noise_power ** 0.5)
    waveform = waveform + noise
    print(f"Added noise (SNR={NOISE_SNR_DB} dB)")

processed, sr = preprocess(waveform=waveform, sr=sr, use_vad=False)
print(f"Audio: {processed.shape[-1]/sr:.2f}s\n")

am = AcousticModel()

# 1. Emissions
emissions = am.get_emissions(processed)

# Save emission matrix visualization
VISUALIZATION_DIR.mkdir(exist_ok=True)
# (T, C) -> plot time x vocab; use prob for interpretable colors
emissions_np = emissions.numpy()
probs = np.exp(emissions_np)  # log_softmax -> prob
T, C = probs.shape
# Order token axis alphabetically by character label
sorted_indices = sorted(range(C), key=lambda i: am.idx_to_char.get(i, "?"))
probs_ordered = probs[:, sorted_indices]
vocab_labels = [am.idx_to_char.get(i, "?") for i in sorted_indices]
fig, ax = plt.subplots(figsize=(14, max(6, C * 0.25)))
im = ax.imshow(probs_ordered.T, aspect="auto", origin="lower", cmap="viridis")
ax.set_xlabel("Time (frame)")
ax.set_ylabel("Token")
ax.set_yticks(range(C))
ax.set_yticklabels(vocab_labels)
ax.set_title(f"Emission matrix (P(token | frame))\n{ref}\nSNR={NOISE_SNR_DB if ADD_NOISE else 'No'} dB")
plt.colorbar(im, ax=ax, label="Probability")
plt.tight_layout()
out_path = VISUALIZATION_DIR / "emission_matrix.png"
plt.savefig(out_path, dpi=150)
plt.close()
print(f"Emission matrix saved to {out_path}")

# 2. Setup the "Normal" LM-backed Decoder
# (This downloads a standard KenLM 4-gram model trained on LibriSpeech)
print("Downloading pretrained LM files (librispeech-4-gram)...")
files = download_pretrained_files("librispeech-4-gram")
print("  done.")

# We need the vocabulary list exactly as wav2vec 2.0 uses it
vocab_list = [am.idx_to_char.get(i, "").lower() for i in range(len(am.vocab))]
vocab_list[am.blank_idx] = "-" # torchaudio expects standard blank symbol
print(f"Vocabulary size: {len(vocab_list)}, blank_idx: {am.blank_idx}")

print("Building CTC decoder (beam_size=500, lm_weight=2.0)...")
decoder = ctc_decoder(
    lexicon=files.lexicon,
    tokens=vocab_list,
    lm=files.lm,
    nbest=3,           # Keep top 3 sentences
    beam_size=500,     # Standard beam width
    lm_weight=15.0,     # The alpha parameter
    word_score=-1.0,   # The beta parameter (insertion penalty/bonus)
    blank_token="-",
    sil_token="|"
)

# 3. Run the decode
# torchaudio expects batch dimension: (B, T, C)
emissions_batched = emissions.unsqueeze(0).cpu()
T = emissions_batched.shape[1]

# Set to True to print hypotheses as decoding progresses (decode on growing prefix)
VISUALIZE_DECODING = True
chunk_frames = max(1, T // 20)  # ~20 steps over the utterance

if VISUALIZE_DECODING:
    print(f"Decoding step-by-step (every ~{chunk_frames} frames)...")
    for end in range(chunk_frames, T + 1, chunk_frames):
        prefix = emissions_batched[:, :end, :]
        step_results = decoder(prefix)
        best = step_results[0][0]
        time_s = end / 50.0  # assume 50 fps for wav2vec2
        am_lm = ""
        if hasattr(best, "am_score") and hasattr(best, "lm_score"):
            am_lm = f"  [am={best.am_score:.2f} lm={best.lm_score:.2f}]"
        print(f"  t={time_s:.2f}s (frame {end:4d}/{T}): {best.words}{am_lm}")
    # Final decode on full emissions
    results = decoder(emissions_batched)
    print("Decode complete.")
else:
    print(f"Decoding (batched shape: {emissions_batched.shape})...")
    results = decoder(emissions_batched)
    print("Decode complete.")

# 4. Extract the best transcript and print all n-best hypotheses
best_hypothesis = results[0][0]  # First item in batch, best hypothesis
print(f"\n--- Result ---")
print(f"Best: {best_hypothesis.words}")
if hasattr(best_hypothesis, "score"):
    print(f"Score: {best_hypothesis.score}")
if hasattr(best_hypothesis, "am_score"):
    print(f"AM score: {best_hypothesis.am_score}")
if hasattr(best_hypothesis, "lm_score"):
    print(f"LM score: {best_hypothesis.lm_score}")

# Print all n-best hypotheses for this utterance
print(f"\n--- Top-{len(results[0])} hypotheses ---")
for i, hyp in enumerate(results[0], 1):
    words_str = " ".join(hyp.words) if isinstance(hyp.words, list) else str(hyp.words)
    parts = [words_str]
    if hasattr(hyp, "score"):
        parts.append(f"score={hyp.score}")
    if hasattr(hyp, "am_score"):
        parts.append(f"am_score={hyp.am_score}")
    if hasattr(hyp, "lm_score"):
        parts.append(f"lm_score={hyp.lm_score}")
    print(f"  {i}. {' | '.join(parts)}")