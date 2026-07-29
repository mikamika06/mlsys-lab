import sys

sys.path.insert(0, ".")
from sched import Scheduler


def _trace(n, plen, olen, gap=0):
    return [{"rid": f"r{i}", "arrival": i * gap, "output_len": olen,
             "prompt": list(range(1000 + i * 100, 1000 + i * 100 + plen))} for i in range(n)]


def _drive(cfg, spec, steps=3000):
    s = Scheduler(dict(cfg))
    s.add(spec)
    peak_running = 0
    peak_blocks = 0
    for _ in range(steps):
        if not s.waiting and not s.running and not s.swapped:
            break
        r = s.step()
        peak_running = max(peak_running, r["running"])
        peak_blocks = max(peak_blocks, r["blocks_used"])
    return s, peak_running, peak_blocks


def test_never_exceeds_max_seqs():
    cfg = {"num_blocks": 64, "max_seqs": 4}
    _, peak, _ = _drive(cfg, _trace(12, 64, 20))
    assert peak <= cfg["max_seqs"], f"{peak} running with max_seqs={cfg['max_seqs']}"


def test_never_exceeds_block_pool():
    cfg = {"num_blocks": 40, "max_seqs": 8}
    _, _, blocks = _drive(cfg, _trace(10, 96, 30))
    assert blocks <= cfg["num_blocks"], f"{blocks} blocks used of {cfg['num_blocks']}"


def test_every_request_terminates():
    cfg = {"num_blocks": 20, "max_seqs": 8, "preemption": "recompute"}
    s, _, _ = _drive(cfg, _trace(8, 16, 200), steps=20000)
    m = s.metrics()
    assert m["finished"] + m["rejected"] == 8, m
    assert not s.running and not s.waiting and not s.swapped


def test_blocks_are_returned():
    cfg = {"num_blocks": 64, "max_seqs": 4}
    s, _, _ = _drive(cfg, _trace(6, 64, 10))
    assert s.alloc.free_count() == cfg["num_blocks"]


def test_shared_block_survives_one_release():
    from sched import Allocator
    a = Allocator(4, 16)
    b = a.allocate()
    a.share(b)
    a.release(b)
    assert a.free_count() == 3, "a block released once while still shared came back to the pool"
    a.release(b)
    assert a.free_count() == 4


def test_free_pool_never_overlaps_live_blocks():
    cfg = {"num_blocks": 96, "max_seqs": 6, "prefix_cache": True}
    shared = list(range(500, 564))
    spec = [{"rid": f"r{i}", "arrival": i, "output_len": 12,
             "prompt": shared + list(range(900 + i * 40, 900 + i * 40 + 48))} for i in range(6)]
    s = Scheduler(dict(cfg))
    s.add(spec)
    for _ in range(3000):
        if not s.waiting and not s.running and not s.swapped:
            break
        s.step()
        live = {b for seq in s.running for b in seq.blocks}
        assert not (live & set(s.alloc.free)), "a live block is sitting in the free pool"
