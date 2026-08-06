import ref


def lookup_error(error_msg):
    for case in ref.TRIAGE_CASES:
        if case["error_msg"].lower() in error_msg.lower() or error_msg.lower() in case["error_msg"].lower():
            return {"cause": case["cause"], "fix": case["fix"]}
    return None
