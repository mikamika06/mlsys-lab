import ref


def check(workdir):
    from vllm_hardening.sanitizer import sanitize_stream_chunk

    out = {"streaming_safe": 0.0}
    ok = 0
    for resp in ref.SAMPLE_RESPONSES:
        cleaned = sanitize_stream_chunk(resp, is_untrusted=True)
        has_leak = False
        def walk(obj):
            nonlocal has_leak
            if isinstance(obj, dict):
                if "reasoning_content" in obj:
                    has_leak = True
                for v in obj.values():
                    walk(v)
            elif isinstance(obj, list):
                for item in obj:
                    walk(item)
        walk(cleaned)
        if not has_leak:
            ok += 1

    if ok == len(ref.SAMPLE_RESPONSES):
        out["streaming_safe"] = 1.0
    return out
