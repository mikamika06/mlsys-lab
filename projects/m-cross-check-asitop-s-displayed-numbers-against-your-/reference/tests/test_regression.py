from edgemetrics.logger import detect_drop

def test_detect_drop_basic():
    samples = [95.0, 96.0, 94.0, 20.0, 95.0]
    idx = detect_drop(samples)
    assert idx == 3
