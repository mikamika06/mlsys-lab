import sys
sys.path.insert(0, ".")

import numpy as np
from sla.profiler import classify_sla_compliance


def test_tail_percentile_violations_are_detected():
    batch_profiles = {
        1: {
            "latencies": [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
            "cpu_time_sec": 1.0,
        },
        4: {
            "latencies": [6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 100.0],
            "cpu_time_sec": 2.0,
        },
    }
    target_sla = {50.0: 15.0, 99.0: 30.0}

    res = classify_sla_compliance(batch_profiles, target_sla)

    assert res["results"][1]["compliant"] is True
    assert res["results"][4]["compliant"] is False
    assert 99.0 in res["results"][4]["violations"]
    assert res["max_compliant_batch"] == 1


def test_all_percentiles_passing():
    batch_profiles = {
        2: {
            "latencies": [10.0] * 100,
            "cpu_time_sec": 0.5,
        }
    }
    target_sla = {50.0: 20.0, 95.0: 20.0, 99.0: 20.0}

    res = classify_sla_compliance(batch_profiles, target_sla)
    assert res["results"][2]["compliant"] is True
    assert len(res["results"][2]["violations"]) == 0
    assert res["max_compliant_batch"] == 2
