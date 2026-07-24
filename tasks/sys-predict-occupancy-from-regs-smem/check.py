def _oracle(regs_per_thread, smem_bytes, threads_per_program, limits):
    warp_size = limits["warp_size"]
    max_warps = limits["max_warps"]
    max_blocks = limits["max_blocks"]
    register_file = limits["register_file"]
    smem_capacity = limits["smem_capacity"]

    warps_per_program = (threads_per_program + warp_size - 1) // warp_size

    blocks_by_regs = register_file // (regs_per_thread * threads_per_program)
    blocks_by_smem = smem_capacity // smem_bytes

    resident_blocks = min(blocks_by_regs, blocks_by_smem, max_blocks)
    return min(max_warps, resident_blocks * warps_per_program)


def grade(sol, fx) -> dict:
    cases = [
        (16, 2048, 128, {
            "warp_size": 32,
            "max_warps": 64,
            "max_blocks": 16,
            "register_file": 65536,
            "smem_capacity": 49152,
        }),
        (64, 8192, 256, {
            "warp_size": 32,
            "max_warps": 64,
            "max_blocks": 8,
            "register_file": 65536,
            "smem_capacity": 49152,
        }),
        (96, 16384, 64, {
            "warp_size": 32,
            "max_warps": 48,
            "max_blocks": 32,
            "register_file": 65536,
            "smem_capacity": 65536,
        }),
        (24, 32768, 512, {
            "warp_size": 32,
            "max_warps": 32,
            "max_blocks": 4,
            "register_file": 131072,
            "smem_capacity": 98304,
        }),
        (40, 4096, 192, {
            "warp_size": 32,
            "max_warps": 80,
            "max_blocks": 20,
            "register_file": 98304,
            "smem_capacity": 65536,
        }),
    ]

    ok = 1.0
    for args in cases:
        try:
            got = sol.predict_occupancy(*args)
        except Exception:
            ok = 0.0
            break
        if got != _oracle(*args):
            ok = 0.0
            break
    return {"exact_match": ok}
