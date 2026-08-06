import ref

def check(workdir):
    from bls_router.router import route_request

    out = {"routing_accuracy": 0.0, "evaluations_matched": 0.0}
    total = len(ref.MOCK_REQUESTS)
    correct_route = 0
    correct_eval = 0

    for req in ref.MOCK_REQUESTS:
        want = ref.reference_bls_route(req)
        got = route_request(req)
        if got.get("branch") == want["branch"]:
            correct_route += 1
        if "result" in got and ref.np.allclose(got["result"], want["result"]):
            correct_eval += 1

    out["routing_accuracy"] = float(correct_route / total) if total > 0 else 0.0
    out["evaluations_matched"] = float(correct_eval / total) if total > 0 else 0.0
    return out
