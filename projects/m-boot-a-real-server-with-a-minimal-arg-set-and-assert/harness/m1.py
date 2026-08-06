import ref


def check(workdir):
    from vllm_boot.config import parse_serve_args

    out = {"args_matched": 0.0}
    ok = True
    for i, tc in enumerate(ref.TEST_CASES):
        try:
            cfg = parse_serve_args(tc["args"])
            if (
                cfg.model != tc["expected_model"]
                or cfg.served_model_name != tc["expected_served_name"]
                or cfg.host != tc["expected_host"]
                or cfg.port != tc["expected_port"]
            ):
                ok = False
                out["_note"] = (
                    f"case {i}: expected model={tc['expected_model']}, served={tc['expected_served_name']}, host={tc['expected_host']}, port={tc['expected_port']}; got model={getattr(cfg, 'model', None)}, served={getattr(cfg, 'served_model_name', None)}, host={getattr(cfg, 'host', None)}, port={getattr(cfg, 'port', None)}"
                )
                break
        except Exception as e:
            ok = False
            out["_note"] = f"case {i} raised {type(e).__name__}: {e}"
            break

    if ok:
        out["args_matched"] = 1.0
    return out
