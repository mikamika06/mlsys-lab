import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from vadd.grid import calculate_launch_waste, cdiv, get_grid_num_programs

    out = {"grid_counts_matched": 0.0, "waste_matched": 0.0}

    grid_ok = True
    waste_ok = True

    for n, block_size in ref.TEST_SIZES:
        want_grid = ref.get_grid_num_programs(n, block_size)
        got_grid = get_grid_num_programs(n, block_size)
        if got_grid != want_grid:
            grid_ok = False
            out["_note"] = f"get_grid_num_programs({n}, {block_size}): got {got_grid}, want {want_grid}"
            break

        want_cdiv = ref.cdiv(n, block_size)
        got_cdiv = cdiv(n, block_size)
        if got_cdiv != want_cdiv:
            grid_ok = False
            out["_note"] = f"cdiv({n}, {block_size}): got {got_cdiv}, want {want_cdiv}"
            break

        want_waste = ref.calculate_launch_waste(n, block_size)
        got_waste = calculate_launch_waste(n, block_size)
        if got_waste != want_waste:
            waste_ok = False
            out["_note"] = f"calculate_launch_waste({n}, {block_size}): got {got_waste}, want {want_waste}"
            break

    if grid_ok:
        out["grid_counts_matched"] = 1.0
    if waste_ok:
        out["waste_matched"] = 1.0

    return out
