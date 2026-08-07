import numpy as np

RECORDED_EAGLE_LOGS = [
    {
        "request_id": "req-alpha",
        "draft_accepted_counts": [3, 4, 2, 5, 1],
        "draft_proposed_counts": [5, 5, 5, 5, 5],
    },
    {
        "request_id": "req-beta",
        "draft_accepted_counts": [1, 2, 0, 1, 0],
        "draft_proposed_counts": [4, 4, 4, 4, 4],
    },
    {
        "request_id": "req-gamma",
        "draft_accepted_counts": [5, 5, 5, 4, 5],
        "draft_proposed_counts": [5, 5, 5, 5, 5],
    },
    {
        "request_id": "req-delta",
        "draft_accepted_counts": [2, 1, 3, 2],
        "draft_proposed_counts": [3, 3, 3, 3],
    },
]

DIAGNOSTIC_CONFIG_RECORDS = [
    {
        "config_id": "cfg-001",
        "speculative_method": "eagle3",
        "num_speculative_tokens": 5,
        "max_model_len_speculative": 16,
        "target_model_arch": "LlamaForCausalLM",
        "draft_model_arch": "LlamaForCausalLM_Eagle3",
        "target_head_dim": 128,
        "draft_head_dim": 128,
        "target_vocab_size": 32000,
        "draft_vocab_size": 32000,
        "expected_outcome": "VALID_EAGLE_CONFIG",
    },
    {
        "config_id": "cfg-002",
        "speculative_method": "speculative_stream",
        "num_speculative_tokens": 4,
        "max_model_len_speculative": 16,
        "target_model_arch": "LlamaForCausalLM",
        "draft_model_arch": "LlamaForCausalLM_Eagle3",
        "target_head_dim": 128,
        "draft_head_dim": 128,
        "target_vocab_size": 32000,
        "draft_vocab_size": 32000,
        "expected_outcome": "ERR_SPEC_METHOD_UNSUPPORTED",
    },
    {
        "config_id": "cfg-003",
        "speculative_method": "eagle3",
        "num_speculative_tokens": 32,
        "max_model_len_speculative": 16,
        "target_model_arch": "LlamaForCausalLM",
        "draft_model_arch": "LlamaForCausalLM_Eagle3",
        "target_head_dim": 128,
        "draft_head_dim": 128,
        "target_vocab_size": 32000,
        "draft_vocab_size": 32000,
        "expected_outcome": "ERR_MAX_DRAFT_EXCEEDED",
    },
    {
        "config_id": "cfg-004",
        "speculative_method": "eagle3",
        "num_speculative_tokens": 4,
        "max_model_len_speculative": 16,
        "target_model_arch": "LlamaForCausalLM",
        "draft_model_arch": "MistralForCausalLM_Eagle3",
        "target_head_dim": 128,
        "draft_head_dim": 128,
        "target_vocab_size": 32000,
        "draft_vocab_size": 32000,
        "expected_outcome": "ERR_DRAFT_MODEL_MISMATCH",
    },
    {
        "config_id": "cfg-005",
        "speculative_method": "eagle3",
        "num_speculative_tokens": 4,
        "max_model_len_speculative": 16,
        "target_model_arch": "LlamaForCausalLM",
        "draft_model_arch": "LlamaForCausalLM_Eagle3",
        "target_head_dim": 128,
        "draft_head_dim": 64,
        "target_vocab_size": 32000,
        "draft_vocab_size": 32000,
        "expected_outcome": "ERR_HEAD_DIM_MISMATCH",
    },
    {
        "config_id": "cfg-006",
        "speculative_method": "eagle3",
        "num_speculative_tokens": 4,
        "max_model_len_speculative": 16,
        "target_model_arch": "LlamaForCausalLM",
        "draft_model_arch": "LlamaForCausalLM_Eagle3",
        "target_head_dim": 128,
        "draft_head_dim": 128,
        "target_vocab_size": 32000,
        "draft_vocab_size": 128256,
        "expected_outcome": "ERR_VOCAB_SIZE_MISMATCH",
    },
]


def reference_compute_request_acceptance(log_records):
    results = []
    for rec in log_records:
        req_id = rec["request_id"]
        tot_acc = sum(rec["draft_accepted_counts"])
        tot_prop = sum(rec["draft_proposed_counts"])
        rate = float(tot_acc / tot_prop) if tot_prop > 0 else 0.0
        results.append({
            "request_id": req_id,
            "total_accepted": tot_acc,
            "total_proposed": tot_prop,
            "mean_acceptance_rate": rate,
        })
    return results


def reference_compute_distribution_summary(request_stats):
    rates = [r["mean_acceptance_rate"] for r in request_stats]
    if not rates:
        return {"p25": 0.0, "p50": 0.0, "p75": 0.0, "mean": 0.0}
    arr = np.array(rates, dtype=np.float64)
    return {
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "mean": float(np.mean(arr)),
    }


def reference_diagnose_speculative_config(cfg):
    method = cfg.get("speculative_method")
    if method != "eagle3":
        return "ERR_SPEC_METHOD_UNSUPPORTED"
    num_spec = cfg.get("num_speculative_tokens", 0)
    max_spec = cfg.get("max_model_len_speculative", 128)
    if num_spec <= 0 or num_spec > max_spec:
        return "ERR_MAX_DRAFT_EXCEEDED"
    target_arch = cfg.get("target_model_arch")
    draft_arch = cfg.get("draft_model_arch")
    if not draft_arch or not draft_arch.startswith(target_arch):
        return "ERR_DRAFT_MODEL_MISMATCH"
    if cfg.get("target_head_dim") != cfg.get("draft_head_dim"):
        return "ERR_HEAD_DIM_MISMATCH"
    if cfg.get("target_vocab_size") != cfg.get("draft_vocab_size"):
        return "ERR_VOCAB_SIZE_MISMATCH"
    return "VALID_EAGLE_CONFIG"
