LOGS = [
    [
        "I0801 model_repository_manager.cc:123] Poll failed for model 'alpha': version 1 is missing file 'model.savedmodel'",
        "I0801 model_repository_manager.cc:123] Poll failed for model 'beta': config.pbtxt missing platform declaration",
        "I0801 model_repository_manager.cc:123] Poll failed for model 'gamma': version 2 directory is empty"
    ],
    [
        "I0801 model_repository_manager.cc:123] Poll failed for model 'delta': version 1 missing config.pbtxt",
        "I0801 model_repository_manager.cc:123] Poll failed for model 'epsilon': invalid version directory name 'v1'"
    ],
    [
        "I0801 model_repository_manager.cc:123] Poll failed for model 'zeta': version 3 missing file 'model.onnx'"
    ]
]

MISMATCH_CASES = [
    {"config": 'name: "m1"\nplatform: "tensorflow_graphdef"\nmax_batch_size: 4', "files": ["model.onnx"], "expected": True},
    {"config": 'name: "m2"\nbackend: "onnxruntime"\nmax_batch_size: 4', "files": ["model.onnx"], "expected": False},
    {"config": 'name: "m3"\nplatform: "tensorrt_plan"\nmax_batch_size: 4', "files": ["model.plan"], "expected": False},
    {"config": 'name: "m4"\nbackend: "pytorch"\nmax_batch_size: 4', "files": ["model.onnx"], "expected": True},
]

def reconstruct_fix(logs):
    fixes = []
    for line in logs:
        if "missing file" in line:
            parts = line.split("'")
            model = parts[1]
            file_name = line.split("missing file ")[1].strip("'")
            fixes.append({"model": model, "action": "create_file", "path": f"{model}/1/{file_name}"})
        elif "missing platform declaration" in line:
            parts = line.split("'")
            model = parts[1]
            fixes.append({"model": model, "action": "update_config", "path": f"{model}/config.pbtxt"})
        elif "directory is empty" in line:
            parts = line.split("'")
            model = parts[1]
            fixes.append({"model": model, "action": "create_dir", "path": f"{model}/2"})
        elif "missing config.pbtxt" in line:
            parts = line.split("'")
            model = parts[1]
            fixes.append({"model": model, "action": "create_file", "path": f"{model}/config.pbtxt"})
        elif "invalid version directory name" in line:
            parts = line.split("'")
            model = parts[1]
            fixes.append({"model": model, "action": "rename_dir", "path": f"{model}/1"})
    return sorted(fixes, key=lambda x: (x["model"], x["action"]))

def detect_mismatch(config_text, filenames):
    backend = None
    platform = None
    for line in config_text.splitlines():
        line = line.strip()
        if line.startswith("backend:"):
            backend = line.split(":")[1].strip().strip('"')
        elif line.startswith("platform:"):
            platform = line.split(":")[1].strip().strip('"')

    has_onnx = any(f.endswith(".onnx") for f in filenames)
    has_plan = any(f.endswith(".plan") or f.endswith(".trt") for f in filenames)
    has_savedmodel = any("savedmodel" in f for f in filenames)
    has_pt = any(f.endswith(".pt") or f.endswith(".pth") for f in filenames)

    if backend == "onnxruntime" and not has_onnx:
        return True
    if platform == "tensorflow_graphdef" and not has_savedmodel:
        return True
    if platform == "tensorrt_plan" and not has_plan:
        return True
    if backend == "pytorch" and not has_pt:
        return True
    if not backend and not platform:
        return True
    return False
