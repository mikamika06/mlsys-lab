import sys
sys.path.insert(0, ".")
from calib.picker import pick_calibration_method
from calib.schema import select_quant_schema

def test_calibration_selection_invariants():
    outlier_stats = {"outlier_ratio": 0.05, "skewness": 2.0, "kl_divergence": 0.8}
    res_outlier = pick_calibration_method(outlier_stats)
    assert res_outlier["method"] == "Percentile", f"expected Percentile, got {res_outlier['method']}"

    clean_stats = {"outlier_ratio": 0.001, "skewness": 0.1, "kl_divergence": 0.05}
    res_clean = pick_calibration_method(clean_stats)
    assert res_clean["method"] == "MinMax", f"expected MinMax, got {res_clean['method']}"

    schema_pos = select_quant_schema(0.0, 10.0, has_zero_point_support=True)
    assert schema_pos["schema"] == "U8S8", f"expected U8S8, got {schema_pos['schema']}"

    schema_sym = select_quant_schema(-5.0, 5.0, has_zero_point_support=True)
    assert schema_sym["schema"] == "S8S8", f"expected S8S8, got {schema_sym['schema']}"
