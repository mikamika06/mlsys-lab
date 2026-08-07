def check(workdir):
    import ref
    from mlx_serve.server import format_chat
    msgs = [{"role": "user", "content": "hello"}]
    res = format_chat(msgs)
    ref_res = ref.apply_chat_template(msgs)
    ok = 1.0 if res.strip() == ref_res.strip() else 0.0
    return {"chat_template_ok": ok}
