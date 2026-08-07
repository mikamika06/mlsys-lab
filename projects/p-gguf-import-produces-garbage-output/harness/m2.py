def check(workdir):
    from gguf_engine.importer import GGUFImporter
    from gguf_engine.tokenizer import GGUFTokenizer
    m = {"template_match": 0.0}
    try:
        imp = GGUFImporter("dummy.gguf")
        tok = GGUFTokenizer(imp.verify_metadata())
        res = tok.apply_chat_template([{"role": "user", "content": "hi"}])
        if "<|user|>" in res:
            m["template_match"] = 1.0
    except Exception:
        pass
    return m
