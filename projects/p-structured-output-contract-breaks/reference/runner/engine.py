import json
import numpy as np
from runner.grammar import JSONGrammarMasker

class ConstrainedEngine:
    def __init__(self, vocab: list, schema: dict):
        self.vocab = vocab
        self.schema = schema
        self.masker = JSONGrammarMasker(schema, vocab)

    def generate(self, logits_generator, num_predict: int) -> dict:
        prefix = ""
        tokens = []
        truncated = False

        for step in range(num_predict):
            raw_logits = logits_generator(prefix)
            masked_logits = self.masker.apply_mask(raw_logits, prefix)

            token_id = int(np.argmax(masked_logits))
            token = self.vocab[token_id]

            prefix += token
            tokens.append(token_id)

            if prefix.endswith("}"):
                try:
                    json.loads(prefix)
                    break
                except Exception:
                    pass
        else:
            truncated = True

        valid = False
        parsed = None
        if not truncated:
            try:
                parsed = json.loads(prefix)
                valid = True
            except Exception:
                valid = False

        return {
            "output": prefix,
            "tokens": tokens,
            "truncated": truncated,
            "valid": valid,
            "parsed": parsed
        }
