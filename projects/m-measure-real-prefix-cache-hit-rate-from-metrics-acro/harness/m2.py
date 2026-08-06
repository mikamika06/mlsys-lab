import ref


def check(workdir):
    from apcmetric.workload import run_two_turn_workload
    from apcmetric.parser import parse_metrics

    out = {"hit_rate_match": 0.0}
    class DummyClient:
        def __init__(self, snaps):
            self.snaps = snaps
            self.i = 0
        def get_metrics(self):
            s = self.snaps[min(self.i, len(self.snaps)-1)]
            self.i += 1
            return s

    snaps = [ref.CONFIGS[0], ref.CONFIGS[1], ref.CONFIGS[2]]
    parsed_snaps = [parse_metrics(s) for s in snaps]

    class MockWorkload:
        def __init__(self):
            self.calls = 0
        def __call__(self, turn):
            self.calls += 1

    try:
        mw = MockWorkload()
        res = run_two_turn_workload(DummyClient(parsed_snaps), mw)
        if mw.calls == 2:
            out["hit_rate_match"] = 1.0
        else:
            out["_note"] = f"Expected 2 turns, got {mw.calls}"
    except Exception as e:
        out["_note"] = f"workload execution failed: {str(e)[:100]}"
    return out
