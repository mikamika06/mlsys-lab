import sys

sys.path.insert(0, ".")
from oaicompat import classify_params, native_counters, shim_counters, hidden_counters
from oaicompat.counters import OPENAI_USAGE_FIELDS

RUNNER_CLEAN = {
    "name": "toy-standard",
    "supported": {"temperature", "top_p"},
    "ignored": set(),
    "native_counters": ["prompt_tokens", "completion_tokens", "total_tokens"],
}

RUNNER_CHATTY = {
    "name": "toy-chatty",
    "supported": {"temperature", "top_p", "seed", "logprobs"},
    "ignored": {"user"},
    "native_counters": [
        "prompt_tokens", "completion_tokens", "total_tokens",
        "cache_hit_tokens", "queue_ms", "gpu_kv_blocks",
    ],
}


def test_every_param_gets_exactly_one_level():
    rows = classify_params(RUNNER_CHATTY)
    names = [r["param"] for r in rows]
    assert len(names) == len(set(names)) == 15
    for r in rows:
        assert r["level"] in ("supported", "ignored", "unsupported")


def test_supported_and_ignored_are_disjoint_outcomes():
    rows = classify_params(RUNNER_CHATTY)
    by_name = {r["param"]: r["level"] for r in rows}
    assert by_name["temperature"] == "supported"
    assert by_name["user"] == "ignored"
    assert by_name["response_format"] == "unsupported"


def test_shim_never_exceeds_the_openai_usage_schema():
    for runner in (RUNNER_CLEAN, RUNNER_CHATTY):
        leaked = set(shim_counters(runner)) - set(OPENAI_USAGE_FIELDS)
        assert not leaked, f"shim exposes non-standard counters: {leaked}"


def test_hidden_counters_recover_everything_the_shim_drops():
    hidden = set(hidden_counters(RUNNER_CHATTY))
    assert hidden == {"cache_hit_tokens", "gpu_kv_blocks", "queue_ms"}
    assert set(native_counters(RUNNER_CLEAN)) == set(shim_counters(RUNNER_CLEAN))
    assert hidden_counters(RUNNER_CLEAN) == []
