import ref
import sys
import os

sys.path.insert(0, os.getcwd())


def check(workdir):
    from ssevall.mapper import map_openai_request

    out = {"params_matched": 0.0}
    match = True
    for req in ref.REQUESTS:
        got = map_openai_request(req)
        if "temperature" in req and got.get("temperature") != float(req["temperature"]):
            match = False
        if "max_tokens" in req and got.get("max_tokens") != int(req["max_tokens"]):
            match = False
        if "extra_body" in req:
            for k, v in req["extra_body"].items():
                if got.get(k) != v:
                    match = False

    if match:
        out["params_matched"] = 1.0
    return out
