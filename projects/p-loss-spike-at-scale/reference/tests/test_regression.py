import sys
sys.path.insert(0, ".")
from loss_spike.detector import SpikeDetector

def test_detector_normal():
    det = SpikeDetector(threshold=3.0)
    losses = [1.0, 1.1, 1.05, 1.08, 1.1]
    for l in losses:
        assert not det.update(l)

def test_detector_spike():
    det = SpikeDetector(threshold=3.0)
    losses = [1.0, 1.0, 1.0, 1.0, 10.0]
    results = [det.update(l) for l in losses]
    assert results[-1] == True
