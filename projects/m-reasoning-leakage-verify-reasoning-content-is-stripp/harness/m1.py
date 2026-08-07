import ref


def check(workdir):
    from vllm_hardening.sanitizer import sanitize_response

    out = {"sanitized_count": 0.0}
    ok = 0
    for resp in ref.SAMPLE_RESPONSES:
        cleaned = sanitize_response(resp, is_untrusted=True)
        def check_no_reasoning(obj):
            if isinstance(obj, dict):
                if "reasoning_content" in obj:
                    return False
                return all(check_no_reasoning(v) for v in obj.values())
            elif isinstance(obj, list):
                return all(check_no_reasoning(item) for item in obj)
            return True

        if check_no_reasoning(cleaned):
            ok += 1

        trusted = sanitize_response(resp, is_untrusted=False)
        if resp != trusted:
            ok = -999

    if ok == len(ref.SAMPLE_RESPONSES):
        out["sanitized_count"] = 1.0
    return out
