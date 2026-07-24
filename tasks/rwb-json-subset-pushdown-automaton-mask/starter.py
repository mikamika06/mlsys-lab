VALUE_TOKENS = {"STR", "NUM", "TRUE", "FALSE", "NULL"}
VALUE_START = VALUE_TOKENS | {"{", "["}


def allowed_next_tokens(prefix: list) -> list:
    """Run the bracket-stack pushdown automaton over `prefix` (a valid
    partial derivation of the JSON-subset grammar) and return the list (or
    set) of token types legally allowed immediately after it. An empty
    prefix or a syntactically complete top-level value are both valid
    inputs (the latter must return an empty list)."""
    raise NotImplementedError('your code here')
