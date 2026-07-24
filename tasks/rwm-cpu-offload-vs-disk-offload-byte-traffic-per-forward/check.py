import random

def _oracle(placement, layer_sizes):
    cpu = 0
    disk = 0
    for layer, loc in placement.items():
        size = layer_sizes[layer]
        if loc == "cpu":
            cpu += size
        elif loc == "disk":
            disk += size
    return (cpu, disk)

def grade(sol, fx) -> dict:
    random.seed(42)
    layer_names = [f"layer_{i}" for i in range(8)]
    cases = []
    for _ in range(5):
        placement = {}
        layer_sizes = {}
        for name in layer_names:
            loc = random.choice(["gpu", "cpu", "disk"])
            placement[name] = loc
            layer_sizes[name] = random.randint(1_000, 100_000_000)
        cases.append((placement, layer_sizes))

    ok = 1.0
    for placement, layer_sizes in cases:
        try:
            got = sol.offload_byte_traffic(placement, layer_sizes)
            got = (int(got[0]), int(got[1]))
        except Exception:
            ok = 0.0
            break
        expected = _oracle(placement, layer_sizes)
        if got != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
