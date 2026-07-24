"""Grade a REAL CUDA-C solve.cu: parse it with arena.cuda_c.CudaProgram and
execute it on the software GPU (arena.cuda_sim.GPU). Compares gmem[0:2]
(number of waves, last-wave utilisation) against a reference computed here
with plain Python arithmetic (no hardcoding, no simulator dependency for the
oracle itself).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlsys.sim import CudaProgram  # noqa: E402
from mlsys.sim import GPU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

TEST_CASES = [
    (8, 2, 4),    # perfect fit: 1 wave, util = 1.0
    (10, 4, 2),   # capacity 8, 2 waves, util = 2/8 = 0.25
    (9, 4, 2),    # capacity 8, 2 waves, util = 1/8 = 0.125
    (33, 8, 4),   # capacity 32, 2 waves, util = 1/32
    (1, 1, 1),    # single block, single SM
    (64, 8, 4),   # capacity 32, 2 full waves, util = 1.0
]


def _reference(num_blocks, num_sms, blocks_per_sm):
    capacity = num_sms * blocks_per_sm
    num_waves = (num_blocks + capacity - 1) // capacity   # ceiling division
    remainder = num_blocks % capacity
    blocks_last = capacity if remainder == 0 else remainder
    last_util = blocks_last / capacity
    return float(num_waves), float(last_util)


def grade(srcfile: str = "solve.cu") -> dict:
    with open(os.path.join(HERE, srcfile)) as f:
        src = f.read()

    try:
        prog = CudaProgram(src)
    except ValueError as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    max_err = 0.0
    for num_blocks, num_sms, bps in TEST_CASES:
        gpu = GPU(2)
        gpu.gmem[:] = -1.0
        params = {"out": 0, "num_blocks": num_blocks, "num_sms": num_sms, "blocks_per_sm": bps}
        try:
            prog.launch(gpu, "wave_calc", 1, 1, params)
        except Exception as e:  # noqa: BLE001 — any runtime fault fails the gate cleanly
            return {"max_abs_err": float("inf"), "error": str(e)}

        ref_w, ref_u = _reference(num_blocks, num_sms, bps)
        got_w, got_u = float(gpu.gmem[0]), float(gpu.gmem[1])
        max_err = max(max_err, abs(got_w - ref_w), abs(got_u - ref_u))

    return {"max_abs_err": max_err}


if __name__ == "__main__":
    import json

    srcfile = sys.argv[1] if len(sys.argv) > 1 else "solve.cu"
    metrics = grade(srcfile)
    meta = json.load(open(os.path.join(HERE, "meta.json")))
    ok = True
    for g in meta["gates"]:
        v = metrics.get(g["metric"])
        gate_ok = v is not None and (
            v <= g["threshold"] if g["op"] == "<=" else
            v >= g["threshold"] if g["op"] == ">=" else v == g["threshold"]
        )
        ok = ok and gate_ok
        print(f"{'PASS' if gate_ok else 'FAIL'}  {g['metric']}={v}  ({g['op']} {g['threshold']})")
    print(json.dumps(metrics, indent=2))
    sys.exit(0 if ok else 1)
