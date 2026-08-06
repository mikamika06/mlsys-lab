import sys
sys.path.insert(0, ".")
from edgequant.variants import build_variants
from edgequant.int8io import export_int8_io
from edgequant.sweep import run_sweep


def test_variants_count_and_types():
    spec = {"w": __import__("numpy").zeros((10, 10), dtype=__import__("numpy").float32)}
    res = build_variants(spec)
    assert len(res) == 4
    assert res["int8_full"]["io_dtype"] == "int8"


def test_int8_io_properties():
    spec = {"w": __import__("numpy").zeros((10, 10), dtype=__import__("numpy").float32)}
    res = export_int8_io(spec)
    assert res["io_dtype"] == "int8"
    assert res["quantized"] is True


def test_sweep_monotone_decrease():
    spec = {"w": __import__("numpy").zeros((10, 10), dtype=__import__("numpy").float32)}
    sizes = [10, 50, 100, 500]
    res = run_sweep(spec, sizes)
    vals = [res[sz] for sz in sizes]
    assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))
