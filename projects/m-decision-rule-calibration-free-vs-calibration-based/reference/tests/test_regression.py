from quantmap.audit import audit_dependencies
from quantmap.rule import build_bpp_table, classify_decision_rule


def test_classify_decision_rule_calibration():
    spec = {
        "has_calibration_data": True,
        "max_calibration_sec": 300,
        "target_bpp": 4.0,
        "accuracy_tolerance": 0.01,
    }
    res = classify_decision_rule(spec)
    assert res["strategy"] == "calibration_based"
    assert res["requires_dataset"] is True


def test_bpp_table_metadata_overhead():
    model_spec = {"total_params": 1_000_000}
    lib_configs = {
        "auto_gptq": {
            "formats": [
                {"name": "gptq_g64", "base_bits": 4, "group_size": 64, "scale_bits": 16, "zero_bits": 16, "deprecated": False}
            ]
        }
    }
    table = build_bpp_table(model_spec, lib_configs)
    assert len(table) == 1
    assert table[0]["bpp"] == 4.5


def test_audit_dependencies_deprecation():
    manifest = {"selected": [{"lib": "auto_gptq", "format": "gptq_act_order_v1"}]}
    registry = {"auto_gptq": {"gptq_act_order_v1": "gptq_g128"}}
    report = audit_dependencies(manifest, registry)
    assert report["valid"] is False
    assert report["deprecated_count"] == 1
