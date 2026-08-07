def check(workdir):
    import ref
    from runner import adapter

    m = {"streaming_ok": 0.0, "tokens_counted": 0.0}
    try:
        chunks = ref.sample_chunks()
        res, tokens = adapter.fix_streaming_and_tokens(chunks)
        if isinstance(res, list) and len(res) > 0:
            m["streaming_ok"] = 1.0
        if isinstance(tokens, int) and tokens > 0:
            m["tokens_counted"] = 1.0
    except Exception:
        pass
    return m
