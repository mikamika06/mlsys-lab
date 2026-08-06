import math

CONFIGS_BREAKEVEN = [
    {"warmup_cost": 15.0, "eager_step": 1.2, "compiled_step": 0.7},
    {"warmup_cost": 45.5, "eager_step": 2.5, "compiled_step": 1.5},
    {"warmup_cost": 10.0, "eager_step": 0.8, "compiled_step": 0.5},
]

def calc_break_even(cfg):
    c = cfg["warmup_cost"]
    te = cfg["eager_step"]
    tc = cfg["compiled_step"]
    saving = te - tc
    if saving <= 0:
        return float('inf')
    return math.ceil(c / saving)

GUARDS_TESTS = [
    {"change": "tensor_shape_change", "triggers_guard": True},
    {"change": "tensor_dtype_change", "triggers_guard": True},
    {"change": "python_global_control_flow", "triggers_guard": True},
    {"change": "optimizer_lr_update_in_place", "triggers_guard": False},
    {"change": "tensor_device_change", "triggers_guard": True},
    {"change": "weight_tensor_data_mutation", "triggers_guard": False},
]
