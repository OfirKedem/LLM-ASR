"""Text normalization utilities for ASR evaluation."""

import re


def normalize_text(text: str) -> str:
    """Lowercase, keep only English letters and spaces, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z ]", "", text)
    text = re.sub(r" +", " ", text).strip()
    return text


def merge_acronyms(text: str) -> str:
    """Merge consecutive single letters separated by spaces.

    "u s a" -> "usa", "n a s a scheduled" -> "nasa scheduled"
    """
    tokens = text.split()
    merged: list[str] = []
    i = 0
    while i < len(tokens):
        if len(tokens[i]) == 1 and tokens[i].isalpha():
            acronym = tokens[i]
            j = i + 1
            while j < len(tokens) and len(tokens[j]) == 1 and tokens[j].isalpha():
                acronym += tokens[j]
                j += 1
            if len(acronym) > 1:
                merged.append(acronym)
            else:
                merged.append(tokens[i])
            i = j
        else:
            merged.append(tokens[i])
            i += 1
    return " ".join(merged)


def normalize_for_eval(text: str) -> str:
    """Full normalization pipeline used before WER/CER comparison."""
    text = normalize_text(text)
    text = merge_acronyms(text)
    return text


# ------------------------------------------------------------------
# Test
# ------------------------------------------------------------------
def test():
    print("=== text_utils test ===\n")

    cases = [
        ("GO DO YOU HEAR", "go do you hear"),
        ("Hello, World! 123", "hello world"),
        ("  multiple   spaces  ", "multiple spaces"),
        ("It's a test.", "its a test"),
    ]
    for raw, expected in cases:
        result = normalize_text(raw)
        status = "PASS" if result == expected else "FAIL"
        print(f"  [{status}] normalize_text({raw!r}) -> {result!r}  (expected {expected!r})")

    print()
    acronym_cases = [
        ("u s a", "usa"),
        ("n a s a scheduled", "nasa scheduled"),
        ("the u s", "the us"),
        ("hello world", "hello world"),
        ("a b c d e", "abcde"),
        ("i went to the u s a today", "i went to the usa today"),
    ]
    for raw, expected in acronym_cases:
        result = merge_acronyms(raw)
        status = "PASS" if result == expected else "FAIL"
        print(f"  [{status}] merge_acronyms({raw!r}) -> {result!r}  (expected {expected!r})")

    print()
    full_cases = [
        ("U S A", "usa"),
        ("N.A.S.A. scheduled the launch", "nasa scheduled the launch"),
        ("GO DO YOU HEAR", "go do you hear"),
    ]
    for raw, expected in full_cases:
        result = normalize_for_eval(raw)
        status = "PASS" if result == expected else "FAIL"
        print(f"  [{status}] normalize_for_eval({raw!r}) -> {result!r}  (expected {expected!r})")

    print("\nDone.")


if __name__ == "__main__":
    test()
