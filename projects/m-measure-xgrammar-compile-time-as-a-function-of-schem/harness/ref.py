import numpy as np

SCHEMAS = [
    {"type": "object", "properties": {"id": {"type": "integer"}}},
    {"type": "object", "properties": {"user": {"type": "object", "properties": {"name": {"type": "string", "pattern": "^[a-z]+$"}}}}},
    {"type": "object", "properties": {"data": {"type": "array", "items": {"type": "number"}}}}
]

VOCAB_SIZE = 32000

def count_nodes(schema):
    if isinstance(schema, dict):
        return 1 + sum(count_nodes(v) for v in schema.values())
    elif isinstance(schema, list):
        return sum(count_nodes(item) for item in schema)
    return 0

def compute_token_mask(vocab_size, allowed_tokens):
    m = np.zeros(vocab_size, dtype=bool)
    m[allowed_tokens] = True
    return m
