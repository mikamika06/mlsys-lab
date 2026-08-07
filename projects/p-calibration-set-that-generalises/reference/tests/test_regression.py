import sys
sys.path.insert(0, ".")
from quant import calibration
import ref

def test_sensitivity_output():
    data = ref.generate_synthetic_data()
    res = calibration.measure_sensitivity(data)
    assert len(res) == 3

def test_domain_comparison():
    data = ref.generate_synthetic_data()
    sens = calibration.measure_sensitivity(data)
    diff = calibration.compare_domains(sens)
    assert diff >= 0.0

def test_min_size_bound():
    data = ref.generate_synthetic_data()
    sz = calibration.find_min_size(data)
    assert sz <= 128

def test_domains_validity():
    data = ref.generate_synthetic_data()
    res = calibration.check_domains(data)
    assert all(res.values())

def test_evaluation_drop_limits():
    data = ref.generate_synthetic_data()
    drops = calibration.evaluate_drop(data)
    assert all(v < 0.05 for v in drops.values())
