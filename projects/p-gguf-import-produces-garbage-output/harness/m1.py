def check(workdir):
    from gguf_engine.importer import GGUFImporter
    m = {"metadata_ok": 0.0}
    try:
        imp = GGUFImporter("dummy.gguf")
        meta = imp.verify_metadata()
        if isinstance(meta, dict) and "architecture" in meta:
            m["metadata_ok"] = 1.0
    except Exception:
        pass
    return m
