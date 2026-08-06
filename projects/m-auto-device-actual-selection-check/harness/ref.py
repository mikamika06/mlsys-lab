import random


def generate_targets(seed=42):
    rng = random.Random(seed)
    devices_pool = ["CPU", "GPU.0", "GPU.1", "NPU"]
    targets = []
    for i in range(10):
        target_id = f"target_{i}"
        hint = rng.choice(["THROUGHPUT", "LATENCY", "DEFAULT"])
        avail = rng.sample(devices_pool, k=rng.randint(1, 3))
        exec_devs = [rng.choice(avail)] if rng.random() > 0.3 else []
        props = {"AVAILABLE_DEVICES": avail}
        if exec_devs:
            props["EXECUTION_DEVICES"] = exec_devs
        
        pref = rng.choice(devices_pool)
        targets.append({
            "id": target_id,
            "device": "AUTO",
            "hint": hint,
            "preferred_device": pref,
            "properties": props
        })
    return targets


def generate_matrix_specs(seed=42):
    rng = random.Random(seed)
    devices = [
        {"name": "CPU", "supports_dynamic": True, "max_dims": 5},
        {"name": "GPU", "supports_dynamic": True, "max_dims": 4},
        {"name": "NPU", "supports_dynamic": False, "max_dims": 4},
        {"name": "VPU", "supports_dynamic": False, "max_dims": 3}
    ]
    shapes = [
        {"name": "static_2d", "is_dynamic": False, "dims": [1, 224]},
        {"name": "static_4d", "is_dynamic": False, "dims": [1, 3, 224, 224]},
        {"name": "dynamic_4d", "is_dynamic": True, "dims": [1, 3, -1, -1]},
        {"name": "static_6d", "is_dynamic": False, "dims": [1, 2, 3, 4, 5, 6]}
    ]
    return devices, shapes
