import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from typing import Union, List

# Load the model and tokenizer 
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval() # Set model to evaluation mode

def get_candidate_probabilities(input_data: Union[str, List[int]], candidate_tokens: List[str]):
    """
    Takes either an input string or a list of token IDs, and a list of candidate token strings.
    Returns the model's predicted probability for each candidate.
    """
    # 1. Handle the input data type (String vs. List of IDs)
    if isinstance(input_data, str):
        # Encode string into tensor of shape [1, sequence_length]
        input_ids = tokenizer.encode(input_data, return_tensors='pt')
    elif isinstance(input_data, list) and all(isinstance(i, int) for i in input_data):
        # Convert list of ints into tensor of shape [1, sequence_length]
        input_ids = torch.tensor([input_data])
    else:
        raise ValueError("input_data must be a string or a list of integer token IDs")

    # 2. Forward pass: get model predictions
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # 3. Get the logits for the very last token in the input sequence
    next_token_logits = logits[0, -1, :]

    # 4. Convert raw logits to probabilities using softmax
    probabilities = torch.nn.functional.softmax(next_token_logits, dim=-1)

    # 5. Extract probabilities for the requested candidates
    results = {}
    for token_text in candidate_tokens:
        token_ids = tokenizer.encode(token_text)
        
        if not token_ids:
            continue
            
        token_id = token_ids[0]
        prob = probabilities[token_id].item()
        results[token_text] = prob

    return results

# --- Example Usage ---
if __name__ == "__main__":
    # Example 1: Using a list of token IDs
    # 1537 = "But", 287 = " in"
    token_id_input = [1537, 287] 
    candidates = [" less", " l"] 
    
    print(f"Input Token IDs: {token_id_input} (Decoded: '{tokenizer.decode(token_id_input)}')")
    probs_ids = get_candidate_probabilities(token_id_input, candidates)
    
    for token, prob in sorted(probs_ids.items(), key=lambda x: x[1], reverse=True):
        print(f"'{token}': {prob:.4%}")
        
    print("-" * 30)

    # Example 2: Using standard text
    text_input = "But in"
    print(f"Input Text: '{text_input}'")
    probs_text = get_candidate_probabilities(text_input, candidates)
    
    for token, prob in sorted(probs_text.items(), key=lambda x: x[1], reverse=True):
        print(f"'{token}': {prob:.4%}")