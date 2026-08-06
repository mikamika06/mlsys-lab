import numpy as np


def compute_perplexity(logits: np.ndarray, targets: np.ndarray) -> float:
    c = np.max(logits, axis=-1, keepdims=True)
    log_sum_exp = c + np.log(np.sum(np.exp(logits - c), axis=-1, keepdims=True))
    log_probs = logits - log_sum_exp
    n_samples = logits.shape[0]
    target_log_probs = log_probs[np.arange(n_samples), targets]
    nll = -np.mean(target_log_probs)
    return float(np.exp(nll))


def compute_kl_divergence(
    teacher_logits: np.ndarray, student_logits: np.ndarray
) -> float:
    c_t = np.max(teacher_logits, axis=-1, keepdims=True)
    log_sum_exp_t = c_t + np.log(
        np.sum(np.exp(teacher_logits - c_t), axis=-1, keepdims=True)
    )
    log_p = teacher_logits - log_sum_exp_t
    p = np.exp(log_p)

    c_s = np.max(student_logits, axis=-1, keepdims=True)
    log_sum_exp_s = c_s + np.log(
        np.sum(np.exp(student_logits - c_s), axis=-1, keepdims=True)
    )
    log_q = student_logits - log_sum_exp_s

    kl_per_sample = np.sum(p * (log_p - log_q), axis=-1)
    return float(np.mean(kl_per_sample))


def rank_quant_candidates(teacher_data: dict, candidates_data: dict) -> dict:
    cand_stats = {}
    for cand_id, cat_map in candidates_data.items():
        kls = []
        ppls = []
        for cat, t_info in teacher_data.items():
            s_logits = cat_map[cat]["logits"]
            t_logits = t_info["logits"]
            targets = t_info["targets"]
            kl = compute_kl_divergence(t_logits, s_logits)
            ppl = compute_perplexity(s_logits, targets)
            kls.append(kl)
            ppls.append(ppl)
        cand_stats[cand_id] = {
            "mean_kl": float(np.mean(kls)),
            "mean_ppl": float(np.mean(ppls)),
        }

    cand_ids = sorted(candidates_data.keys())

    sorted_by_kl = sorted(cand_ids, key=lambda c: (cand_stats[c]["mean_kl"], c))
    sorted_by_ppl = sorted(
        cand_ids, key=lambda c: (cand_stats[c]["mean_ppl"], c)
    )

    kl_ranks = {c: i + 1 for i, c in enumerate(sorted_by_kl)}
    ppl_ranks = {c: i + 1 for i, c in enumerate(sorted_by_ppl)}

    out_candidates = {}
    for c in cand_ids:
        out_candidates[c] = {
            "mean_kl": cand_stats[c]["mean_kl"],
            "mean_ppl": cand_stats[c]["mean_ppl"],
            "kl_rank": kl_ranks[c],
            "ppl_rank": ppl_ranks[c],
        }

    disagreement = sorted_by_kl[0] != sorted_by_ppl[0]

    return {"candidates": out_candidates, "rank_disagreement": disagreement}


def detect_contaminated_baseline(
    teacher_data: dict,
    historical_ppl: dict,
    max_perplexity_drop_ratio: float = 0.4,
    min_entropy_threshold: float = 1e-3,
) -> dict:
    flagged = []
    cat_stats = {}

    for cat in sorted(teacher_data.keys()):
        data = teacher_data[cat]
        logits = data["logits"]
        targets = data["targets"]

        ppl = compute_perplexity(logits, targets)

        c = np.max(logits, axis=-1, keepdims=True)
        log_sum_exp = c + np.log(
            np.sum(np.exp(logits - c), axis=-1, keepdims=True)
        )
        log_p = logits - log_sum_exp
        p = np.exp(log_p)

        entropies = -np.sum(p * log_p, axis=-1)
        mean_entropy = float(np.mean(entropies))

        hist_ppl = historical_ppl.get(cat, ppl)
        ppl_drop = ppl < (1.0 - max_perplexity_drop_ratio) * hist_ppl
        low_ent = mean_entropy < min_entropy_threshold

        if ppl_drop and low_ent:
            reason = "both"
        elif ppl_drop:
            reason = "perplexity_drop"
        elif low_ent:
            reason = "low_entropy"
        else:
            reason = "ok"

        is_flagged = ppl_drop or low_ent
        if is_flagged:
            flagged.append(cat)

        cat_stats[cat] = {
            "perplexity": float(ppl),
            "entropy": float(mean_entropy),
            "flagged": is_flagged,
            "reason": reason,
        }

    return {
        "is_contaminated": len(flagged) > 0,
        "flagged_categories": sorted(flagged),
        "category_stats": cat_stats,
    }


def evaluate_acceptance_gate(
    candidate_metrics: dict, category_thresholds: dict
) -> dict:
    cat_results = {}
    failed = []

    for cat in sorted(category_thresholds.keys()):
        thresh = category_thresholds[cat]
        metrics = candidate_metrics.get(
            cat, {"kl": float("inf"), "ppl": float("inf")}
        )
        kl_pass = metrics["kl"] <= thresh["max_kl"]
        ppl_pass = metrics["ppl"] <= thresh["max_ppl"]
        passed = kl_pass and ppl_pass

        cat_results[cat] = {
            "kl_pass": bool(kl_pass),
            "ppl_pass": bool(ppl_pass),
            "passed": bool(passed),
        }

        if not passed:
            failed.append(cat)

    return {
        "accepted": len(failed) == 0,
        "category_results": cat_results,
        "failed_categories": sorted(failed),
    }


def get_m1_test_cases():
    rng = np.random.RandomState(42)
    categories = ["code", "math", "chat"]

    teacher_data = {}
    candidates_data = {"int8_weight_only": {}, "fp8_e4m3": {}, "int4_awq": {}}

    for cat in categories:
        N, C = 40, 10
        t_logits = rng.randn(N, C) * 2.0
        targets = rng.randint(0, C, size=N)
        teacher_data[cat] = {"logits": t_logits, "targets": targets}

        c1_logits = t_logits + rng.randn(N, C) * 0.1

        c2_logits = t_logits.copy()
        c2_logits[np.arange(N), targets] += 3.0
        c2_logits += rng.randn(N, C) * 0.8

        c3_logits = t_logits + rng.randn(N, C) * 0.5

        candidates_data["int8_weight_only"][cat] = {"logits": c1_logits}
        candidates_data["fp8_e4m3"][cat] = {"logits": c2_logits}
        candidates_data["int4_awq"][cat] = {"logits": c3_logits}

    return teacher_data, candidates_data


def get_m2_contamination_cases():
    rng = np.random.RandomState(123)

    teacher_data = {}
    historical_ppl = {
        "cat_clean": 8.5,
        "cat_ppl_leak": 15.0,
        "cat_low_ent": 15.0,
        "cat_both": 15.0,
    }

    N, C = 30, 8
    clean_logits = rng.randn(N, C) * 0.1
    clean_targets = rng.randint(0, C, size=N)
    teacher_data["cat_clean"] = {"logits": clean_logits, "targets": clean_targets}

    leak_logits = rng.randn(N, C) * 0.5
    leak_targets = rng.randint(0, C, size=N)
    leak_logits[np.arange(N), leak_targets] += 8.0
    teacher_data["cat_ppl_leak"] = {"logits": leak_logits, "targets": leak_targets}

    low_ent_logits = np.zeros((N, C))
    low_ent_logits[:, 0] = 100.0
    low_ent_targets = rng.randint(1, C, size=N)
    teacher_data["cat_low_ent"] = {
        "logits": low_ent_logits,
        "targets": low_ent_targets,
    }

    both_targets = rng.randint(0, C, size=N)
    both_logits = np.zeros((N, C))
    both_logits[np.arange(N), both_targets] = 100.0
    teacher_data["cat_both"] = {"logits": both_logits, "targets": both_targets}

    return teacher_data, historical_ppl


def get_m2_gate_cases():
    candidate_metrics = {
        "code": {"kl": 0.02, "ppl": 10.5},
        "math": {"kl": 0.08, "ppl": 14.2},
        "chat": {"kl": 0.01, "ppl": 8.1},
    }

    thresh_pass = {
        "code": {"max_kl": 0.05, "max_ppl": 12.0},
        "math": {"max_kl": 0.10, "max_ppl": 15.0},
        "chat": {"max_kl": 0.05, "max_ppl": 10.0},
    }

    thresh_fail = {
        "code": {"max_kl": 0.05, "max_ppl": 9.0},
        "math": {"max_kl": 0.05, "max_ppl": 15.0},
        "chat": {"max_kl": 0.05, "max_ppl": 10.0},
    }

    return candidate_metrics, thresh_pass, thresh_fail
