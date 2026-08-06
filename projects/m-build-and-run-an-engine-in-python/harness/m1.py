import ref

def check(workdir):
    from engine.parser import parse_model, ParseError
    ok = 0
    total = len(ref.MODELS)
    for model in ref.MODELS:
        try:
            parse_model(model, ref.SUPPORTED_OPS)
        except ParseError:
            ok += 1
        except Exception:
            pass
        else:
            if all(n["op"] in ref.SUPPORTED_OPS for n in model["nodes"]):
                ok += 1
    passed = 1.0 if ok == total else 0.0
    return {"parser_matched": passed}
