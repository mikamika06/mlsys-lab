import re
from typing import List

# Reference pattern identical to the official GPT‑2 pre‑tokenizer regex
_PATTERN = r"(?:(?:[^\x00-\x7F]+)|(?:\w+)|(?:[^\s\w]))"

def _ref_split(text: str) -> List[str]:
    """Compute the reference split using the official regex."""
    return re.findall(_PATTERN, text)

def grade(sol, fx) -> dict:
    """
    Grade a candidate solution.

    The grader runs a set of representative strings through both the
    candidate implementation and the reference implementation.  It then
    compares the resulting lists for exact equality.
    """
    cases = [
        "Hello world!",
        "Café",
        "123 456",
        "foo-bar_baz",
        "😀 emoji 😃",
        "",
        "   ",
        "a b c d e f g h i j k l m n o p q r s t u v w x y z",
    ]
    for text in cases:
        try:
            got = sol.split_gpt2_pre_tokenizer(text)
        except Exception:
            return {"exact_match": 0.0}
        ref = _ref_split(text)
        if got != ref:
            return {"exact_match": 0.0}
    return {"exact_match": 1.0}
