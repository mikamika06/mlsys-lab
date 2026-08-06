import sys

sys.path.insert(0, ".")
from pagecache.estimator import estimate_resident_bytes


def test_unaligned_boundary_residency():
    file_size = 100000
    accesses = [(100, 50)]
    got = estimate_resident_bytes(file_size, accesses, page_size=4096)
    assert got == 4096, f"Expected 4096 bytes for single page fault, got {got}"


def test_straddling_page_boundary():
    file_size = 100000
    accesses = [(4000, 200)]
    got = estimate_resident_bytes(file_size, accesses, page_size=4096)
    assert got == 8192, f"Expected 8192 bytes for straddled access, got {got}"
