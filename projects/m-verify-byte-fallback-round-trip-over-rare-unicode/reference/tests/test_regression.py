import sys
sys.path.insert(0, ".")
from bytefallback.convert import encode_with_fallback, decode_with_fallback, verify_round_trip

VOCAB = {
    "hello": 1,
    "world": 2,
    " ": 3,
}
for i in range(256):
    VOCAB[f"<0x{i:02X}>"] = 100 + i

INV_VOCAB = {v: k for k, v in VOCAB.items()}


def test_standard_and_rare_unicode_round_trip():
    samples = [
        "hello world",
        "𓀀𓀁𓀂",
        "𠮷野家",
        "👨‍👩‍👧‍👦",
        "hello 𓀀 world",
    ]
    for sample in samples:
        assert verify_round_trip(sample, VOCAB), f"Failed round-trip for {sample!r}"


def test_byte_aggregation_recovers_multibyte_utf8():
    text = "𓀀"
    encoded = encode_with_fallback(text, VOCAB)
    assert len(encoded) == 4
    for tid in encoded:
        assert INV_VOCAB[tid].startswith("<0x")
    decoded = decode_with_fallback(encoded, INV_VOCAB)
    assert decoded == text
