"""Checker for Milestone 2: PromQL generation and alert evaluation."""

import ref


def check(workdir):
    out = {"promql_queries_valid": 0.0, "alerts_evaluated_correctly": 0.0}

    try:
        from vllm_obs.promql import (
            get_kv_utilization_query,
            get_p95_ttft_query,
            get_preemptions_per_minute_query,
            get_waiting_queue_saturation_query,
        )
        from vllm_obs.alerting import evaluate_alerts
    except ImportError as e:
        out["_note"] = f"Failed to import vllm_obs modules: {e}"
        return out

    try:
        q1 = get_p95_ttft_query("5m")
        q2 = get_kv_utilization_query()
        q3 = get_waiting_queue_saturation_query()
        q4 = get_preemptions_per_minute_query("5m")

        if (
            "histogram_quantile" in q1
            and "vllm:time_to_first_token_seconds" in q1
            and "vllm:gpu_cache_usage_perc" in q2
            and "vllm:num_requests_waiting" in q3
            and "rate" in q4
            and "vllm:num_preemptions_total" in q4
        ):
            out["promql_queries_valid"] = 1.0
        else:
            out["_note"] = "PromQL queries missing key functions or metric names"
    except Exception as e:
        out["_note"] = f"PromQL query generation failed: {e}"
        return out

    alerts_ok = True
    for test_case in ref.TEST_SNAPSHOTS:
        try:
            res = evaluate_alerts(test_case["input"], test_case["thresholds"])
            if len(res) != test_case["expected_alert_count"]:
                alerts_ok = False
                out["_note"] = (
                    f"Expected {test_case['expected_alert_count']} alerts, got {len(res)}"
                )
                break
        except Exception as e:
            alerts_ok = False
            out["_note"] = f"evaluate_alerts failed: {e}"
            break

    if alerts_ok:
        out["alerts_evaluated_correctly"] = 1.0

    return out
