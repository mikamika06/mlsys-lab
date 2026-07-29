import sys

sys.path.insert(0, ".")
from coldpath import build_phases, classify_request, cumulative_ms, total_wall_clock

EVENTS = [
    {"name": "request_in", "t": 0},
    {"name": "daemon_start", "t": 5},
    {"name": "daemon_end", "t": 800},
    {"name": "load_start", "t": 800},
    {"name": "load_end", "t": 3200},
    {"name": "prefill_start", "t": 3200},
    {"name": "prefill_end", "t": 3650},
    {"name": "decode_start", "t": 3650},
    {"name": "decode_end", "t": 4400},
    {"name": "request_out", "t": 4420},
]
BY_NAME = {e["name"]: e["t"] for e in EVENTS}


def test_no_phase_merges_two_named_intervals():
    for p in build_phases(EVENTS):
        want_start = BY_NAME[p["name"] + "_start"]
        want_end = BY_NAME[p["name"] + "_end"]
        assert p["start"] == want_start, f"{p['name']} start {p['start']} != {want_start}"
        assert p["end"] == want_end, f"{p['name']} end {p['end']} != {want_end}"


def test_every_present_pair_becomes_its_own_phase():
    names = [p["name"] for p in build_phases(EVENTS)]
    assert names == ["daemon", "load", "prefill", "decode"], names
    assert len(names) == len(set(names)), "a phase name repeats"


def test_phase_durations_never_exceed_wall_clock():
    total = total_wall_clock(EVENTS)
    attributed = sum(p["duration_ms"] for p in build_phases(EVENTS))
    assert attributed <= total, f"phases claim {attributed}ms inside a {total}ms request"
    assert classify_request(EVENTS) == "cold"


def test_timeline_never_goes_backwards():
    checkpoints = list(range(0, 4500, 50))
    s = cumulative_ms(EVENTS, checkpoints)
    assert all(s[i] <= s[i + 1] for i in range(len(s) - 1)), "attributed time decreased"
    assert s[0] == 0
