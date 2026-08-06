def is_valid_run(run_data):
    for req in run_data:
        if len(req.get("timestamps", [])) < 2:
            return False
        ts = req["timestamps"]
        if any(ts[i] > ts[i+1] for i in range(len(ts)-1)):
            return False
        if req.get("stalled", False):
            return False
    return True
