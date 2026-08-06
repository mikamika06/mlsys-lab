def print_tir_loop_nest(tir_mod):
    lines = []
    indent = 0
    loops = tir_mod.get("loops", ["i", "j", "k"])
    for loop in loops:
        prefix = "  " * indent
        if "[par]" in loop:
            clean = loop.replace("[par]", "")
            lines.append(f"{prefix}for {clean} in parallel(0, ...):")
        elif "[vec]" in loop:
            clean = loop.replace("[vec]", "")
            lines.append(f"{prefix}for {clean} in vectorize(0, ...):")
        elif "_outer" in loop:
            lines.append(f"{prefix}for {loop} in split_outer(0, ...):")
        elif "_inner" in loop:
            lines.append(f"{prefix}for {loop} in split_inner(0, ...):")
        else:
            lines.append(f"{prefix}for {loop} in reorder(0, ...):")
        indent += 1
    lines.append("  " * indent + "C[i, j] += A[i, k] * B[k, j]")
    return "\n".join(lines)


def map_axes_to_transforms(loop_nest_str):
    mapping = {}
    lines = loop_nest_str.strip().splitlines()
    for line in lines:
        line_str = line.strip()
        if not line_str.startswith("for "):
            continue
        parts = line_str.split()
        axis_name = parts[1]
        if "parallel" in line_str:
            mapping[axis_name] = "parallel"
        elif "vectorize" in line_str:
            mapping[axis_name] = "vectorize"
        elif "split_outer" in line_str:
            mapping[axis_name] = "split_outer"
        elif "split_inner" in line_str:
            mapping[axis_name] = "split_inner"
        elif "reorder" in line_str:
            mapping[axis_name] = "reorder"
    return mapping
