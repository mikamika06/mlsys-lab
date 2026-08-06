"""Reference data generator and mock utilities."""

VOCAB = {
    0: "{",
    1: '"a"',
    2: ":",
    3: '"b"',
    4: "}",
    5: ",",
    6: "<EOS>",
    7: "0",
    8: "true",
}
EOS_ID = 6

SCHEMAS = [
    {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
    },
    {
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        "required": ["a"],
    },
    {
        "type": "object",
        "properties": {"a": {"type": "boolean"}},
        "required": ["a"],
    },
    {
        "type": "object",
        "properties": {
            "a": {"type": "string"},
            "b": {"type": "string"},
        },
        "required": ["a", "b"],
    },
]


def mock_logits_fn(tokens):
    import numpy as np

    rng = np.random.RandomState(len(tokens))
    return list(rng.randn(len(VOCAB)))
