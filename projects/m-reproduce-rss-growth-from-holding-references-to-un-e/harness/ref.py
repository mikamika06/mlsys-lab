"""Reference oracle logic and benchmark parameters for harness validation."""

STEPS = 15
ARRAY_SIZE = 512 * 1024
BASE_RSS = 20 * 1024 * 1024
BYTES_PER_ELEM = 4

GRAPH_SPEC = {"num_ops": 4, "shape": (64, 128)}
SHAPE_SEQUENCE = [(16, 64), (16, 64), (32, 64), (16, 64), (64, 64), (32, 64)]


def expected_retained_rss(num_steps, array_size):
    elem_bytes = array_size * BYTES_PER_ELEM
    out = []
    for step in range(1, num_steps + 1):
        chain_bytes = step * elem_bytes
        out.append(BASE_RSS + chain_bytes)
    return out


def expected_recompile_counts(shape_sequence):
    seen = set()
    recompile_cnt = 0
    cached_cnt = 0
    for shp in shape_sequence:
        key = tuple(shp)
        if key not in seen:
            seen.add(key)
            recompile_cnt += 1
        else:
            cached_cnt += 1
    return recompile_cnt, cached_cnt
