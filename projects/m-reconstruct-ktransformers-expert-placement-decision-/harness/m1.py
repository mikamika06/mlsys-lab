import sys
import os

def check(workdir):
    sys.path.insert(0, workdir)
    import ref
    from ktrans.placement import reconstruct_placement

    cases = ref.generate_placement_cases()
    matched = 0
    for i, c in enumerate(cases):
        want = ref.reconstruct_placement(
            c["num_layers"], c["num_experts"], c["expert_bytes"], c["vram_budget"], c["frequency_log"]
        )
        got = reconstruct_placement(
            c["num_layers"], c["num_experts"], c["expert_bytes"], c["vram_budget"], c["frequency_log"]
        )
        if got == want:
            matched += 1

    return {"placements_matched": float(matched)}
