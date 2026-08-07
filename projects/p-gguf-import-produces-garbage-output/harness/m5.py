def check(workdir):
    from gguf_engine.importer import GGUFImporter
    from gguf_engine.tokenizer import GGUFTokenizer
    from gguf_engine.engine import GGUFEngine
    m = {"dialogue_ok": 0.0}
    try:
        imp = GGUFImporter("dummy.gguf")
        tok = GGUFTokenizer(imp.verify_metadata())
        engine = GGUFEngine(imp, tok)
        msgs = [{"role": "user", "content": f"turn {i}"} for i in range(6)]
        out = engine.generate(msgs)
        if isinstance(out, str) and len(out) > 0:
            m["dialogue_ok"] = 1.0
    except Exception:
        pass
    return m
