import ref

def check(workdir):
    from triton_grid.kernel import launch_grid_config

    out = {"kernel_outputs_matched": 0.0}
    try:
        cfg = ref.CONFIGS[0]
        got = launch_grid_config(cfg["width"], cfg["height"], cfg["block_w"], cfg["block_h"])
        gx = math_ceil(cfg["width"], cfg["block_w"]) if "math_ceil" in globals() else __import__("math").ceil(cfg["width"] / cfg["block_w"])
        gy = __import__("math").ceil(cfg["height"] / cfg["block_h"])
        if got.get("grid_x") == gx and got.get("grid_y") == gy and got.get("total_programs") == gx * gy:
            out["kernel_outputs_matched"] = 1.0
    except Exception as e:
        out["_note"] = str(e)
    return out

def math_ceil(a, b):
    return __import__("math").ceil(a / b)
