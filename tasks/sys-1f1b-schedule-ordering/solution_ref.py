def generate_1f1b_schedule(stages, microbatches):
    schedule = []

    for stage in range(stages):
        warmup = stages - stage - 1
        ops = []
        fwd = 0
        bwd = 0

        while fwd < warmup:
            ops.append(f"F{fwd}")
            fwd += 1

        while fwd < microbatches:
            ops.append(f"F{fwd}")
            fwd += 1
            if bwd < fwd - warmup:
                ops.append(f"B{bwd}")
                bwd += 1

        while bwd < microbatches:
            ops.append(f"B{bwd}")
            bwd += 1

        schedule.append(ops)

    return schedule
