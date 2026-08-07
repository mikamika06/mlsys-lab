def check(workdir):
    from gguf_engine.tokenizer import GGUFTokenizer
    from gguf_engine.importer import GGUFImporter
    m = {"stops_ok": 0.0}
    try:
        imp = GGUFImporter("dummy.gguf")
        tok = GGUFTokenizer(imp.verify_metadata())
        stops = tok.get_stop_sequences()
        if isinstance(stops, list) and len(stops) > 0:
            m["stops_ok"] = 1.0
    except Exception:
        pass
    return m
