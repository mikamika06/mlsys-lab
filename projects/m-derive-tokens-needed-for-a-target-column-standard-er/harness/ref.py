import os
import tempfile


def generate_test_cases():
    cases = [
        {"variance": 2.5, "target_se": 0.1},
        {"variance": 10.0, "target_se": 0.5},
        {"variance": 0.0, "target_se": 1.0},
        {"variance": 5.123, "target_se": 0.05}
    ]
    return cases


def generate_dat_fixture():
    fd, path = tempfile.mkstemp(suffix=".dat")
    os.close(fd)
    with open(path, "wb") as f:
        f.write(b"IMATRIX_DUMMY_PAYLOAD_BYTES")
    return path
