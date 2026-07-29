import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from mlsys.sim.server import Server, ServerConfig, make_requests  # noqa: E402

SHARED = tuple(range(500, 564))


def trace(n, plen, olen, gap, prefix=()):
    out = []
    for i in range(n):
        body = list(range(1000 + i * 100, 1000 + i * 100 + plen - len(prefix)))
        out.append({"rid": f"r{i}", "arrival": i * gap, "output_len": olen,
                    "prompt": list(prefix) + body})
    return out


def oracle(spec, cfg):
    s = Server(ServerConfig(**cfg))
    m = s.run(make_requests([dict(d) for d in spec]))
    return m, s.trace


def learner(workdir, spec, cfg):
    from sched import Scheduler
    s = Scheduler(dict(cfg))
    s.add([dict(d) for d in spec])
    m = s.run()
    return m, getattr(s, "trace", [])


CFG_BASE = dict(num_blocks=256, max_seqs=8)
CFG_TIGHT = dict(num_blocks=20, max_seqs=8, preemption="recompute")
CFG_SWAP = dict(num_blocks=40, max_seqs=4, preemption="swap", swap_blocks=64)
CFG_CACHE = dict(num_blocks=256, max_seqs=8, prefix_cache=True)
