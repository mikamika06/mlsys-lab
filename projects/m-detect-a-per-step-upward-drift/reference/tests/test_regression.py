from leak.drift import detect_drift

def test_drift_detection_positive():
    series = [100, 105, 110, 115, 120, 125]
    res = detect_drift(series)
    assert res["has_drift"] is True

def test_drift_detection_negative():
    series = [100, 100, 100, 100, 100, 100]
    res = detect_drift(series)
    assert res["has_drift"] is False
