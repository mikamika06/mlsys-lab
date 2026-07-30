import sys

sys.path.insert(0, ".")
from diskplan import best_native_scheme, disk_size, size_table, tensor_bytes

SCHEMES = [
    {"name": "W16A16", "bits": 16, "group_size": None},
    {"name": "W8A16", "bits": 8, "group_size": 0},
    {"name": "W4A16", "bits": 4, "group_size": 128},
    {"name": "W4A4", "bits": 4, "group_size": 32},
]

MODEL = {"tensors": [
    {"name": "embed", "kind": "embed", "count": 4096},
    {"name": "lin", "kind": "linear", "count": 100000},
    {"name": "norm", "kind": "norm", "count": 256},
]}

HARDWARE = {"native_bits": [16, 8]}


def test_embed_and_norm_never_shrink():
    for scheme in SCHEMES:
        assert tensor_bytes({"kind": "embed", "count": 4096}, scheme) == 8192
        assert tensor_bytes({"kind": "norm", "count": 256}, scheme) == 512


def test_every_scheme_appears_once_in_order():
    table = size_table(MODEL, SCHEMES)
    assert [row["scheme"] for row in table] == [s["name"] for s in SCHEMES]
    assert len(table) == len(SCHEMES)


def test_large_linear_tensor_quantizes_smaller_than_fp16():
    fp16 = disk_size(MODEL, SCHEMES[0])
    for scheme in SCHEMES[1:]:
        assert disk_size(MODEL, scheme) < fp16, f"{scheme['name']} did not shrink the checkpoint"


def test_best_native_never_none_when_fp16_is_native():
    assert best_native_scheme(MODEL, HARDWARE, SCHEMES) is not None
