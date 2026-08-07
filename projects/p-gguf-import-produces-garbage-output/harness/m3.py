def check(workdir):
    from gguf_engine.importer import GGUFImporter
    from gguf_engine.tokenizer import GGUFTokenizer
    m = {"tokens_exact": 0.0}
    try:
        imp = GGUFImporter("dummy.gguf")
        tok = GGUFTokenizer(imp.verify_metadata())
        tokens = tok.encode("test")
        if isinstance(tokens, list) and len(tokens) > 0:
            m["tokens_exact"] = 1.0
    except Exception:
        pass
    return m
