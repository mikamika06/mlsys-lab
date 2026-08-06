CONFIGS = [
    {"layers": 32, "vram_limit_mb": 4096, "layer_memory_mb": 100},
    {"layers": 48, "vram_limit_mb": 8192, "layer_memory_mb": 150},
    {"layers": 24, "vram_limit_mb": 2048, "layer_memory_mb": 120},
]


def compute_reference_ngl(cfg):
    best_ngl = 0
    max_tps = 0.0
    layers = cfg["layers"]
    vram = cfg["vram_limit_mb"]
    mem = cfg["layer_memory_mb"]
    for ngl in range(layers + 1):
        if ngl * mem > vram:
            break
        tps = (ngl * 15.0) + ((layers - ngl) * 2.0)
        if tps > max_tps:
            max_tps = tps
            best_ngl = ngl
    return best_ngl


TENSOR_TEST_CASES = [
    (512, 512),
    (256, 127),
    (1024, 768),
    (100, 256),
]
