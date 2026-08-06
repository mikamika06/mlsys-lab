import ref


def check(workdir):
    from vllm_boot.config import parse_serve_args
    from vllm_boot.server import ServerInstance, boot_and_query_models

    out = {"server_booted": 0.0, "models_endpoint_matched": 0.0}

    try:
        cfg = parse_serve_args(ref.TEST_CASES[0]["args"])
        srv = ServerInstance(cfg)
        srv.boot()
        if srv.is_running:
            out["server_booted"] = 1.0
    except Exception as e:
        out["_note"] = f"boot failed: {type(e).__name__}: {e}"
        return out

    ok_endpoint = True
    for i, tc in enumerate(ref.TEST_CASES):
        try:
            res = boot_and_query_models(tc["args"])
            if (
                not isinstance(res, dict)
                or res.get("object") != "list"
                or "data" not in res
                or not isinstance(res["data"], list)
                or len(res["data"]) < 1
                or res["data"][0].get("id") != tc["expected_served_name"]
            ):
                ok_endpoint = False
                out["_note"] = (
                    f"case {i}: expected endpoint id={tc['expected_served_name']}, got response={res}"
                )
                break
        except Exception as e:
            ok_endpoint = False
            out["_note"] = (
                f"case {i} endpoint query failed: {type(e).__name__}: {e}"
            )
            break

    if ok_endpoint:
        out["models_endpoint_matched"] = 1.0

    return out
