import numpy as np


class SchemaMasker:
    """Compiles a JSON schema into a state machine token mask."""

    def __init__(self, schema: dict, vocab: list[str]):
        self.schema = schema
        self.vocab = vocab
        self.vocab_map = {tok: i for i, tok in enumerate(vocab)}
        self.eos_token = "<eos>"
        self.transitions = {}
        self.terminal_states = set()
        self._compile()

    def _add_trans(self, src: int, tok_str: str, dst: int):
        if src not in self.transitions:
            self.transitions[src] = {}
        if tok_str in self.vocab_map:
            tid = self.vocab_map[tok_str]
            self.transitions[src][tid] = dst

    def _compile(self):
        stype = self.schema.get("type")
        if stype == "enum":
            term = 2
            self.terminal_states.add(term)
            for val in self.schema.get("values", []):
                self._add_trans(0, str(val), 1)
            self._add_trans(1, self.eos_token, term)
        elif stype == "boolean":
            term = 2
            self.terminal_states.add(term)
            self._add_trans(0, "true", 1)
            self._add_trans(0, "false", 1)
            self._add_trans(1, self.eos_token, term)
        elif stype == "object":
            props = self.schema.get("properties", {})
            curr = 0
            self._add_trans(curr, "{", curr + 1)
            curr += 1
            prop_items = list(props.items())
            for i, (k, sub) in enumerate(prop_items):
                key_str = f'"{k}"'
                self._add_trans(curr, key_str, curr + 1)
                curr += 1
                self._add_trans(curr, ":", curr + 1)
                curr += 1
                vtype = sub.get("type")
                if vtype == "boolean":
                    self._add_trans(curr, "true", curr + 1)
                    self._add_trans(curr, "false", curr + 1)
                    curr += 1
                elif vtype == "enum":
                    for val in sub.get("values", []):
                        self._add_trans(curr, str(val), curr + 1)
                    curr += 1
                elif vtype == "number":
                    for d in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                        self._add_trans(curr, d, curr + 1)
                    curr += 1
                if i < len(prop_items) - 1:
                    self._add_trans(curr, ",", curr + 1)
                    curr += 1
            self._add_trans(curr, "}", curr + 1)
            curr += 1
            term = curr + 1
            self._add_trans(curr, self.eos_token, term)
            self.terminal_states.add(term)

    def get_mask(self, state: int) -> np.ndarray:
        mask = np.zeros(len(self.vocab), dtype=bool)
        allowed = self.transitions.get(state, {})
        for tid in allowed:
            mask[tid] = True
        return mask

    def next_state(self, state: int, token_id: int) -> int:
        return self.transitions.get(state, {}).get(token_id, -1)

    def is_terminal(self, state: int) -> bool:
        return state in self.terminal_states


def compile_schema(schema: dict, vocab: list[str]) -> SchemaMasker:
    """Compiles a small JSON schema into a SchemaMasker instance."""
    return SchemaMasker(schema, vocab)
