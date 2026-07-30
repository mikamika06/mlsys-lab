def classify_operation(op):
    if op["runs_untrusted_code"] or op["needs_sandbox_proximity"]:
        return "runner"
    if op["needs_durable_state"]:
        return "server"
    return "client"


def classify_all(config):
    return [{"name": op["name"], "side": classify_operation(op)} for op in config["operations"]]
