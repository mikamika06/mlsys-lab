def classify_operation(op):
    if op["runs_untrusted_code"] or op["needs_sandbox_proximity"]:
        return "runner"
    if op["needs_durable_state"]:
        return "server"
    return "client"


def classify_all(config):
    return [{"name": op["name"], "side": classify_operation(op)} for op in config["operations"]]


def side_sequence(config, order):
    sides = {op["name"]: classify_operation(op) for op in config["operations"]}
    return [sides[name] for name in order]


def hop_count(config, order):
    seq = side_sequence(config, order)
    return sum(1 for a, b in zip(seq, seq[1:]) if a != b)


def is_legal_pipeline(config, order):
    seq = side_sequence(config, order)
    forbidden = {("client", "runner"), ("runner", "client")}
    return not any((a, b) in forbidden for a, b in zip(seq, seq[1:]))


def _op(name, untrusted, sandbox, durable):
    return {"name": name, "runs_untrusted_code": untrusted, "needs_sandbox_proximity": sandbox, "needs_durable_state": durable}


FLAGSHIP = {"operations": [
    _op("parse_cli_arguments", False, False, False),
    _op("render_progress_output", False, False, False),
    _op("read_local_skeleton_file", False, False, False),
    _op("enqueue_grading_job", False, False, True),
    _op("persist_job_status", False, False, True),
    _op("route_logs_to_waiting_client", False, False, True),
    _op("execute_learner_solution", True, False, False),
    _op("import_and_call_learner_module", True, False, False),
    _op("enforce_process_timeout", False, True, False),
    _op("measure_peak_child_memory", False, True, False),
]}

TRUTH_TABLE = {"operations": [
    _op(f"t_{i:03b}", bool(i & 4), bool(i & 2), bool(i & 1)) for i in range(8)
]}

SINGLETON = {"operations": [_op("noop_client_only", False, False, False)]}

SHARED_FLAGS = {"operations": [
    _op("a1", False, False, False),
    _op("a2", False, False, False),
    _op("b1", False, False, True),
    _op("b2", False, False, True),
    _op("c1", True, False, False),
    _op("c2", True, False, False),
]}

CASES = [FLAGSHIP, TRUTH_TABLE, SINGLETON, SHARED_FLAGS]

PIPELINES = [
    {"config": FLAGSHIP, "order": [
        "parse_cli_arguments", "read_local_skeleton_file", "enqueue_grading_job",
        "persist_job_status", "execute_learner_solution", "enforce_process_timeout",
        "import_and_call_learner_module", "measure_peak_child_memory",
        "persist_job_status", "route_logs_to_waiting_client", "render_progress_output",
    ]},
    {"config": FLAGSHIP, "order": ["parse_cli_arguments", "execute_learner_solution", "render_progress_output"]},
    {"config": TRUTH_TABLE, "order": [f"t_{i:03b}" for i in range(8)]},
    {"config": TRUTH_TABLE, "order": ["t_000", "t_100"]},
]
