"""Language model: GPT-2 wrapper with token filtering, top-K, and KV caching."""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from config import Config


class LanguageModel:
    def __init__(self, model_name: str | None = None, device: str | None = None):
        cfg = Config()
        model_name = model_name or cfg.lm_model_name
        self.device = device or cfg.device

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.eos_token_id = self.tokenizer.eos_token_id
        self.space_token_id = self.tokenizer.encode(" ")[0]
        self.vocab_size = self.tokenizer.vocab_size

        # Pre-compute mask of valid (alphabetic + space only) token ids
        self.valid_mask = self._build_valid_mask()
        self.valid_token_ids = self.valid_mask.nonzero(as_tuple=True)[0].tolist()

    # ------------------------------------------------------------------ #
    #  Token filtering                                                    #
    # ------------------------------------------------------------------ #
    def _build_valid_mask(self) -> torch.Tensor:
        """
        Boolean mask: True for tokens whose text is purely composed of
        english alphabetic characters (a-z, A-Z), space, or apostrophe (').
        """
        allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ '")
        mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        for idx in range(self.vocab_size):
            decoded = self.tokenizer.decode([idx])
            if decoded and all(c in allowed_chars for c in decoded):
                mask[idx] = True
        # Always keep EOS so the decoder can terminate
        mask[self.eos_token_id] = True
        mask[self.space_token_id] = False
        return mask

    # ------------------------------------------------------------------ #
    #  Top-K with KV cache                                                #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def top_k(
        self,
        prefix_ids: list[int],
        k: int,
        past_key_values=None,
    ) -> tuple[torch.Tensor, torch.Tensor, object]:
        """Return (token_ids, log_probs, new_kv_cache) for top-*k* valid tokens.

        If *past_key_values* is provided only the last token of *prefix_ids*
        is fed through the model (KV cache reuse).
        """
        if past_key_values is not None and len(prefix_ids) > 0:
            input_ids = torch.tensor([[prefix_ids[-1]]], device=self.device)
        else:
            if len(prefix_ids) == 0:
                input_ids = torch.tensor([[self.eos_token_id]], device=self.device)
            else:
                input_ids = torch.tensor([prefix_ids], device=self.device)
            past_key_values = None

        out = self.model(input_ids=input_ids, past_key_values=past_key_values)
        logits = out.logits[:, -1, :]  # (1, V)
        new_kv = out.past_key_values

        # Mask invalid tokens
        inv_mask = ~self.valid_mask.to(logits.device)
        logits[:, inv_mask] = float("-inf")

        log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)  # (V,)
        topk = torch.topk(log_probs, k=min(k, len(self.valid_token_ids)))
        return topk.indices, topk.values, new_kv

    @torch.no_grad()
    def top_k_from_text(
        self,
        prefix_text: str,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (token_ids, log_probs) for top-*k* valid tokens given text prefix.

        Tokenizes *prefix_text* and runs the model on the full sequence (no KV cache).
        """
        prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
        ids, log_probs, _ = self.top_k(prefix_ids=prefix_ids, k=k, past_key_values=None)
        return ids, log_probs

    def decode_token(self, token_id: int) -> str:
        """Decode a single token id to its text representation."""
        return self.tokenizer.decode([token_id])


# ------------------------------------------------------------------
# Test
# ------------------------------------------------------------------
def test():
    print("=== language_model test ===\n")

    lm = LanguageModel()
    n_valid = len(lm.valid_token_ids)
    print(f"Total vocab     : {lm.vocab_size}")
    print(f"Valid tokens    : {n_valid}")
    sample_tokens = [lm.decode_token(tid) for tid in lm.valid_token_ids[:20]]
    print(f"Sample valid    : {sample_tokens}\n")

    # 1. Top-K with empty prefix
    ids, lps, kv = lm.top_k(prefix_ids=[], k=10)
    print("Top-10 (empty prefix):")
    for tid, lp in zip(ids.tolist(), lps.tolist()):
        print(f"  {lm.decode_token(tid)!r:15s}  log_prob={lp:.4f}")

    # 2. Top-K given "go do you" -> expect "hear" somewhere
    prefix = lm.tokenizer.encode("go do you")
    print(f"\nPrefix tokens for 'go do you': {prefix}")
    ids2, lps2, kv2 = lm.top_k(prefix_ids=prefix, k=10)
    print("Top-10 (prefix='go do you'):")
    decoded_top = []
    for tid, lp in zip(ids2.tolist(), lps2.tolist()):
        tok = lm.decode_token(tid)
        decoded_top.append(tok.strip().lower())
        print(f"  {tok!r:15s}  log_prob={lp:.4f}")

    if "hear" in decoded_top:
        print("  -> 'hear' found in top-10!")
    else:
        all_ids, all_lps, _ = lm.top_k(prefix_ids=prefix, k=5000)
        all_decoded = [lm.decode_token(t).strip().lower() for t in all_ids.tolist()]
        if "hear" in all_decoded:
            rank = all_decoded.index("hear") + 1
            print(f"  -> 'hear' found at rank {rank} in top-5000")
        else:
            print("  -> 'hear' NOT found in top-5000")

    # 3. KV cache consistency
    print("\nKV cache test:")
    prefix_short = lm.tokenizer.encode("go do")
    _, _, kv_cached = lm.top_k(prefix_ids=prefix_short, k=1)
    prefix_full = lm.tokenizer.encode("go do you")
    ids_no_cache, lps_no_cache, _ = lm.top_k(prefix_ids=prefix_full, k=5)
    ids_cached, lps_cached, _ = lm.top_k(
        prefix_ids=prefix_full, k=5, past_key_values=kv_cached
    )
    match = torch.allclose(lps_no_cache, lps_cached, atol=1e-4)
    print(f"  Log-probs match (cache vs no-cache): {match}")
    if not match:
        print(f"  no-cache: {lps_no_cache.tolist()}")
        print(f"  cached  : {lps_cached.tolist()}")

    print("\nDone.")


if __name__ == "__main__":
    test()
