import re
from typing import List

# Official GPT‑2 pre‑tokenizer regex pattern
_PATTERN = r"(?:(?:[^\x00-\x7F]+)|(?:\w+)|(?:[^\s\w]))"

def split_gpt2_pre_tokenizer(text: str) -> List[str]:
    """
    Return the list of pre‑tokens produced by GPT‑2's regex.

    Parameters
    ----------
    text : str
        The raw input string to be tokenised.

    Returns
    -------
    List[str]
        A list of strings, each representing a pre‑token in order.
    """
    return re.findall(_PATTERN, text)
