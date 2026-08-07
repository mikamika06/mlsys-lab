import ref


def check(workdir):
    from vllm_pred.predictor import predict_backend
    from vllm_pred.rejection import get_rejection_reason

    out = {"priority_match": 0.0, "rejection_match": 0.0}
    p_ok = 0
    r_ok = 0
    total = len(ref.LOGS)
    for log_text in ref.LOGS:
        parsed = ref.parse_log(log_text)
        eval_b = parsed["evaluated"]
        rej = parsed["rejections"]

        want_pred = ref.predict_backend(eval_b, rej)
        got_pred = predict_backend(eval_b, rej)
        if got_pred == want_pred:
            p_ok += 1

        b_test = eval_b[0] if eval_b else "FLASH_ATTN"
        want_rej = ref.get_rejection_reason(b_test, rej)
        got_rej = get_rejection_reason(b_test, rej)
        if got_rej == want_rej:
            r_ok += 1

    out["priority_match"] = 1.0 if p_ok == total else 0.0
    out["rejection_match"] = 1.0 if r_ok == total else 0.0
    return out
