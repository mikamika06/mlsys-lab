DEVICE_TEST_CASES = [
    {"is_built": True, "is_available": True},
    {"is_built": True, "is_available": False},
    {"is_built": False, "is_available": False},
]

TIMING_TRACES = [
    [5.0, 5.2, 4.8],
    [10.1, 10.2, 10.0],
]

CPU_REFERENCE_VECTORS = [
    [1.5, 2.5, 3.5],
    [10.0, 20.0, 30.0],
]

MPS_REFERENCE_VECTORS = [
    [1.50001, 2.50001, 3.50001],
    [10.0001, 20.0001, 30.0001],
]
