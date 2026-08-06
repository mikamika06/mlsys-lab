import effbpw.compute

def test_f32_norms():
    shapes = {
        "attention.weight": (1024, 1024),
        "norm.weight": (1024,)
    }
    got = effbpw.compute.compute_effective_bpw(shapes, 4.0)
    assert abs(got - 4.0) > 1e-4, "Norms should raise the effective BPW above the base BPW"
    assert got > 4.0
