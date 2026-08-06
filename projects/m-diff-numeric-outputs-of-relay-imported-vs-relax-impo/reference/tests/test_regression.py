import numpy as np
from migration.diff import compare_relay_relax_outputs
from migration.artifact import profile_artifact_sizes


def test_numerical_discrepancy_bound():
    spec = {"seed": 100, "size": 32}
    err = compare_relay_relax_outputs(spec)
    assert err < 0.01


def test_artifact_size_monotonicity():
    spec = {"seed": 100, "size": 32}
    sizes = profile_artifact_sizes(spec)
    assert 0 in sizes
    assert 2 in sizes
    assert 3 in sizes
    assert sizes[0] >= sizes[2]
    assert sizes[2] >= sizes[3]
