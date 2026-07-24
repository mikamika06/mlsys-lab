import re
from typing import List

# TODO: this implementation incorrectly splits only word characters,
# missing punctuation and non‑ASCII tokens.
_PATTERN = r"\\w+"

def split_gpt2_pre_tokenizer(text: str) -> List[str]:
    """
    A broken attempt at the GPT‑2 pre‑tokeniser.

    It uses a simplistic regex that matches only contiguous word
    characters, so it fails to capture punctuation, spaces and
    non‑ASCII sequences.
    """
    return re.findall(_PATTERN, text)
