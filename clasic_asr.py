from torchaudio.models.decoder import ctc_decoder
from torchaudio.models.decoder import download_pretrained_files

from acoustic_model import AcousticModel
from test_utils import load_test_sample
from preprocessing import preprocess


# 1. Get the emissions from your existing class

print("=== acoustic_model test ===\n")

waveform, sr, ref = load_test_sample(folder="61", chapter="70968", idx=0)
print(f"Reference: {ref}")
processed, sr = preprocess(waveform=waveform, sr=sr, use_vad=False)
print(f"Audio: {processed.shape[-1]/sr:.2f}s\n")

am = AcousticModel()

# 1. Emissions
emissions = am.get_emissions(processed)

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
    lm_weight=20.0,     # The alpha parameter
    word_score=-1.0,   # The beta parameter (insertion penalty/bonus)
    blank_token="-",
    sil_token="|"
)

# 3. Run the decode
# torchaudio expects batch dimension: (B, T, C)
emissions_batched = emissions.unsqueeze(0).cpu()
print(f"Decoding (batched shape: {emissions_batched.shape})...")

# Run the beam search + LM shallow fusion
results = decoder(emissions_batched)
print("Decode complete.")

# 4. Extract the best transcript
best_hypothesis = results[0][0] # First item in batch, best hypothesis
print(f"\n--- Result ---")
print(f"Normal ASR Output: {best_hypothesis.words}")
if hasattr(best_hypothesis, "score"):
    print(f"Score: {best_hypothesis.score}")