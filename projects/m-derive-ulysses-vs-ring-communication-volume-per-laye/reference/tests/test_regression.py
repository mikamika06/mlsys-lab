import pytest
from seqcomm.formulas import (
    ulysses_comm_volume_per_layer,
    ring_comm_volume_per_layer,
    usp_comm_volume_per_layer,
)
from seqcomm.sweep import sweep_comm_costs


def test_regression_sequence_parallelism():
    """Verify sequence parallelism formulas and invariants."""
    N, D, P, H = 8192, 4096, 8, 32

    u_vol = ulysses_comm_volume_per_layer(N, D, P)
    r_vol = ring_comm_volume_per_layer(N, D, P)
    usp_pure_u = usp_comm_volume_per_layer(N, D, P, ulysses_degree=P, ring_degree=1)
    usp_pure_r = usp_comm_volume_per_layer(N, D, P, ulysses_degree=1, ring_degree=P)

    assert u_vol == usp_pure_u, "USP with ring_degree=1 must equal pure Ulysses volume"
    assert r_vol == usp_pure_r, "USP with ulysses_degree=1 must equal pure Ring volume"

    with pytest.raises(ValueError):
        usp_comm_volume_per_layer(N, D, P, ulysses_degree=4, ring_degree=4)

    results = sweep_comm_costs([4096], [2048], [4], [8])
    assert len(results) == 1
    assert results[0]["ulysses_valid"] is True
    assert len(results[0]["usp_configs"]) > 0
