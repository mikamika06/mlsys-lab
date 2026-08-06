import random


def generate_test_cases():
    rng = random.Random(42)
    dtypes = ["float32", "float16", "bfloat16"]
    shapes = [(8, 32), (16, 32), (32, 64), (64, 128)]

    stream = []
    for _ in range(30):
        shape = rng.choice(shapes)
        dtype = rng.choice(dtypes)
        s0 = shape[1] if rng.random() > 0.3 else shape[1] * 2
        s1 = 1
        is_contig = (s0 == shape[1])
        stream.append({
            "shape": shape,
            "dtype": dtype,
            "strides": (s0, s1),
            "is_contiguous": is_contig
        })

    def compile_fn(meta):
        return {
            "guards": [
                {"type": "shape", "dim": 0, "val": meta["shape"][0]},
                {"type": "shape", "dim": 1, "val": meta["shape"][1]},
                {"type": "dtype", "val": meta["dtype"]},
                {"type": "stride", "dim": 0, "val": meta["strides"][0]},
                {"type": "contiguous", "val": meta["is_contiguous"]}
            ]
        }

    return stream, compile_fn
