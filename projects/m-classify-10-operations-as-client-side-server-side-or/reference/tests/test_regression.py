import sys

sys.path.insert(0, ".")
from opside import classify_all, hop_count, is_legal_pipeline

CONFIG = {"operations": [
    {"name": "parse_cli_arguments", "runs_untrusted_code": False, "needs_sandbox_proximity": False, "needs_durable_state": False},
    {"name": "enqueue_grading_job", "runs_untrusted_code": False, "needs_sandbox_proximity": False, "needs_durable_state": True},
    {"name": "execute_learner_solution", "runs_untrusted_code": True, "needs_sandbox_proximity": False, "needs_durable_state": False},
    {"name": "enforce_process_timeout", "runs_untrusted_code": False, "needs_sandbox_proximity": True, "needs_durable_state": False},
    {"name": "render_progress_output", "runs_untrusted_code": False, "needs_sandbox_proximity": False, "needs_durable_state": False},
]}
BY_NAME = {op["name"]: op for op in CONFIG["operations"]}


def test_untrusted_or_sandboxed_operation_never_leaves_the_runner():
    for item in classify_all(CONFIG):
        op = BY_NAME[item["name"]]
        if op["runs_untrusted_code"] or op["needs_sandbox_proximity"]:
            assert item["side"] == "runner", f"{item['name']} needs isolation but landed on {item['side']}"


def test_every_operation_lands_in_exactly_one_side():
    items = classify_all(CONFIG)
    names = [item["name"] for item in items]
    assert sorted(names) == sorted(BY_NAME), f"{sorted(names)} != {sorted(BY_NAME)}"
    assert all(item["side"] in ("client", "server", "runner") for item in items)


def test_pipeline_never_lets_client_talk_to_runner_directly():
    order = ["parse_cli_arguments", "execute_learner_solution", "render_progress_output"]
    assert is_legal_pipeline(CONFIG, order) is False, "client->runner->client should be illegal"


def test_hop_count_never_negative_and_bounded():
    order = [op["name"] for op in CONFIG["operations"]]
    h = hop_count(CONFIG, order)
    assert 0 <= h <= len(order) - 1
