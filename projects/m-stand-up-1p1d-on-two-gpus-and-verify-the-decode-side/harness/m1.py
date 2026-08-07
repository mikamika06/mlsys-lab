import ref


def check(workdir):
    try:
        from disagg.p1d import DecodeWorker, Pipeline1P1D, PrefillWorker
    except Exception as e:
        return {"requests_handled": 0.0, "kv_transferred": 0.0, "_note": f"Import failure: {e}"}

    pw = PrefillWorker(0, 4, 32, 4)
    dw = DecodeWorker(1, 4, 32, 4)
    pipe = Pipeline1P1D(pw, dw)

    handled = 0
    transferred = 0
    total = len(ref.TEST_REQUESTS)

    for req in ref.TEST_REQUESTS:
        try:
            res = pipe.process_request(req["id"], req["prompt"], req["steps"])
            if res and len(res.get("tokens", [])) == req["steps"]:
                handled += 1
            if req["id"] in dw.kv_store:
                transferred += 1
        except Exception as e:
            return {"requests_handled": 0.0, "kv_transferred": 0.0, "_note": f"Execution error: {e}"}

    return {
        "requests_handled": float(handled == total),
        "kv_transferred": float(transferred == total)
    }
