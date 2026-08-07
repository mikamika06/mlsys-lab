import math
from swapspace.sizing import compute_block_bytes


def simulate_preemption_trajectory(config, events):
    blk = compute_block_bytes(config)
    allocated_blocks = 0
    active_seqs = {}
    trajectory = []
    peak_bytes = 0

    for ev in events:
        etype = ev["type"]
        sid = ev["seq_id"]
        tokens = ev["tokens"]
        if etype == "swap_out":
            n_blocks = math.ceil(tokens / config["block_size"])
            active_seqs[sid] = n_blocks
            allocated_blocks += n_blocks
        elif etype == "swap_in":
            if sid in active_seqs:
                allocated_blocks -= active_seqs.pop(sid)

        curr_bytes = allocated_blocks * blk
        trajectory.append(curr_bytes)
        if curr_bytes > peak_bytes:
            peak_bytes = curr_bytes

    return {"trajectory": trajectory, "peak_bytes": peak_bytes}
