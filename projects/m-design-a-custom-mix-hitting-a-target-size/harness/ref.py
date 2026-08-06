CONFIGS = [
    {
        "tensors": [
            {"name": "tensor_a", "numel": 1000, "shape": [10, 100], "ftype_map": {"Q4_0": "Q4_0", "F32": "F32"}},
            {"name": "tensor_b", "numel": 50, "shape": [50], "ftype_map": {"Q4_0": "F32", "F32": "F32"}}
        ],
        "target_bytes": 4000,
        "options": ["Q4_0", "F32"],
        "max_bytes": 5000,
        "allowed_quants": ["Q4_0", "F32"]
    },
    {
        "tensors": [
            {"name": "tensor_c", "numel": 2000, "shape": [20, 100], "ftype_map": {"Q8_0": "Q8_0", "F32": "F32"}},
            {"name": "tensor_d", "numel": 100, "shape": [100], "ftype_map": {"Q8_0": "F32", "F32": "F32"}}
        ],
        "target_bytes": 8000,
        "options": ["Q8_0", "F32"],
        "max_bytes": 10000,
        "allowed_quants": ["Q8_0", "F32"]
    },
    {
        "tensors": [
            {"name": "tensor_e", "numel": 500, "shape": [5, 100], "ftype_map": {"Q4_0": "Q4_0", "F32": "F32"}},
            {"name": "tensor_f", "numel": 20, "shape": [20], "ftype_map": {"Q4_0": "F32", "F32": "F32"}}
        ],
        "target_bytes": 2000,
        "options": ["Q4_0", "F32"],
        "max_bytes": 3000,
        "allowed_quants": ["Q4_0", "F32"]
    }
]

def design_mix(tensors, target_bytes, options):
    return ref.design_mix(tensors, target_bytes, options)

def verify_1d_tensors(tensors, ftypes):
    return ref.verify_1d_tensors(tensors, ftypes)

def choose_quant_under_budget(tensors, max_bytes, allowed_quants):
    return ref.choose_quant_under_budget(tensors, max_bytes, allowed_quants)

import ref
