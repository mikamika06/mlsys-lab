SCENARIOS = [
    {
        "id": "dynamo_guard_fail",
        "layer": "dynamo",
        "code_snippet": "def f(x): return x + 1 if x.sum() > 0 else x - 1",
        "decomposition_count": 0,
        "repro_code": "import torch\n\n@torch.compile\ndef f(x):\n    return x + 1 if x.sum() > 0 else x - 1\n\nx = torch.randn(4)\nf(x)"
    },
    {
        "id": "aot_decomposition_fail",
        "layer": "aot",
        "code_snippet": "def f(x): return torch.special.bessel_j0(x)",
        "decomposition_count": 14,
        "repro_code": "import torch\n\n@torch.compile(backend=\"aot_eager\")\ndef f(x):\n    return torch.special.bessel_j0(x)\n\nx = torch.randn(4)\nf(x)"
    },
    {
        "id": "inductor_codegen_fail",
        "layer": "inductor",
        "code_snippet": "def f(x): return torch.ops.aten._unsafe_index.Tensor(x, [])",
        "decomposition_count": 5,
        "repro_code": "import torch\n\n@torch.compile(backend=\"inductor\")\ndef f(x):\n    return torch.ops.aten._unsafe_index.Tensor(x, [])\n\nx = torch.randn(4)\nf(x)"
    }
]

def identify_layer(scenario_id):
    for s in SCENARIOS:
        if s["id"] == scenario_id:
            return s["layer"]
    return "unknown"

def extract_repro(scenario_id):
    for s in SCENARIOS:
        if s["id"] == scenario_id:
            return s["repro_code"]
    return ""

def count_decompositions(scenario_id):
    for s in SCENARIOS:
        if s["id"] == scenario_id:
            return s["decomposition_count"]
    return 0
