"""Regression tests for TP quantization layouts."""
import numpy as np
from tpquant.checker import diagnose_config
from tpquant.layout import analyze_tp_slice
from tpquant.parallel import prepare_tp_linear


def test_desc_act_tp4_detected_as_unsafe():
    rng = np.random.RandomState(42)
    perm = rng.permutation(128)
    analysis = analyze_tp_slice(128, 32, 4, perm)
    assert not analysis["is_safe"]
    assert len(analysis["oob_ranks"]) > 0 or analysis["fragmented_groups_count"] > 0


def test_validate_only_raises_on_unsafe_layout():
    rng = np.random.RandomState(42)
    perm = rng.permutation(128)
    scales = rng.randn(4, 64)
    try:
        prepare_tp_linear(128, 64, 32, 4, perm, scales, "validate_only")
        assert False
    except ValueError:
        pass


def test_replicate_scales_produces_full_scale_views():
    rng = np.random.RandomState(42)
    perm = rng.permutation(128)
    scales = rng.randn(4, 64)
    res = prepare_tp_linear(128, 64, 32, 4, perm, scales, "replicate_scales")
    assert res["mode"] == "replicate_scales"
    assert len(res["ranks"]) == 4
    for r in res["ranks"]:
        assert r["scales"].shape == (4, 64)
        assert len(r["g_idx"]) == 32


def test_diagnose_config_recommends_fix():
    diag = diagnose_config(4096, 32, 4, True)
    assert diag["has_oob"] is True
    assert diag["has_fragmentation"] is True
    assert diag["recommended_mode"] == "replicate_scales"
