import numpy as np

def get_leak_fixtures():
    np.random.seed(42)
    fixtures = []
    for i in range(5):
        base = 1000 + i * 100
        if i % 2 == 0:
            snaps = [{"step": j, "active_bytes": base + j * 50} for j in range(5)]
            expected = True
        else:
            snaps = [{"step": j, "active_bytes": base + (50 if j % 2 == 0 else -30)} for j in range(5)]
            expected = False
        fixtures.append({"snapshots": snaps, "expected": expected})
    return fixtures

def get_checkpoint_fixtures():
    np.random.seed(42)
    fixtures = []
    for i in range(5):
        layers = [int(x) for x in np.random.randint(100, 500, size=4)]
        batch_size = 2
        normal = sum(layers) * batch_size
        ckpt = max(layers) * batch_size + (sum(layers) // 2)
        expected = normal - ckpt
        fixtures.append({"layers": layers, "batch_size": batch_size, "expected": expected})
    return fixtures

def get_allocator_fixtures():
    np.random.seed(42)
    fixtures = []
    for i in range(5):
        types = np.random.choice(["split", "segment"], size=20, p=[0.4, 0.6])
        events = [{"type": t} for t in types]
        splits = sum(1 for e in events if e["type"] == "split")
        total = len(events)
        expected = float(splits) / float(total) if total > 0 else 0.0
        fixtures.append({"events": events, "expected": expected})
    return fixtures
