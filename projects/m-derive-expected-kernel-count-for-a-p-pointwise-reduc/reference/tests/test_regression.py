from kernelplan.derivation import derive_kernel_count

def test_regression():
    assert derive_kernel_count(3, True, 2) <= 3
    assert derive_kernel_count(0, False, 0) == 0
    assert derive_kernel_count(5, True, 4) >= 2
