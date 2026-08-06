from streamstat.detect import is_valid_run

def test_valid_run():
    good = [{"timestamps": [0.1, 0.2, 0.3], "stalled": False}]
    assert is_valid_run(good) is True

def test_invalid_timestamps():
    bad = [{"timestamps": [0.3, 0.2, 0.4], "stalled": False}]
    assert is_valid_run(bad) is False

def test_stalled_run():
    bad = [{"timestamps": [0.1, 0.2], "stalled": True}]
    assert is_valid_run(bad) is False
