import ref

def check(workdir):
    out = {"classes_registered": 0.0}
    try:
        from ggufconv.converter import _REGISTRY, SynthModelConverter
        if "synth_tiny" in _REGISTRY:
            out["classes_registered"] = 1.0
        else:
            out["_note"] = "synth_tiny not found in registry"
    except Exception as e:
        out["_note"] = str(e)[:120]
    return out
