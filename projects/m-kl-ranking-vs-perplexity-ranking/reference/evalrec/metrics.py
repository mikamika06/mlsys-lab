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
