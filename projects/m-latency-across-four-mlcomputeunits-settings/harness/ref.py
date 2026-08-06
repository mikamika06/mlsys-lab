import numpy as np

COMPUTE_UNITS = ("cpu_only", "cpu_and_gpu", "all", "cpu_and_ne")


def generate_test_graph():
    return {
        "transfer_cost": 2.0,
        "nodes": [
            {
                "name": "stem_conv",
                "supported_units": ["cpu_only", "gpu", "ane"],
                "op_latency": {"cpu": 10.0, "gpu": 4.0, "ane": 1.5},
            },
            {
                "name": "body_block_1",
                "supported_units": ["cpu_only", "gpu", "ane"],
                "op_latency": {"cpu": 20.0, "gpu": 8.0, "ane": 3.0},
            },
            {
                "name": "custom_eval_op",
                "supported_units": ["cpu_only"],
                "op_latency": {"cpu": 5.0},
            },
            {
                "name": "head_conv",
                "supported_units": ["cpu_only", "gpu", "ane"],
                "op_latency": {"cpu": 12.0, "gpu": 5.0, "ane": 2.0},
            },
        ],
    }


def compute_units_latency(graph_profile, compute_units):
    total_time = 0.0
    current_unit = None
    transfer_cost = graph_profile.get("transfer_cost", 1.5)

    for op in graph_profile.get("nodes", []):
        supported = op.get("supported_units", ["cpu_only"])
        op_time = op.get("op_latency", {})

        if compute_units == "all":
            if "ane" in supported:
                assigned = "ane"
            elif "gpu" in supported:
                assigned = "gpu"
            else:
                assigned = "cpu"
        elif compute_units == "cpu_and_ne":
            if "ane" in supported:
                assigned = "ane"
            else:
                assigned = "cpu"
        elif compute_units == "cpu_and_gpu":
            if "gpu" in supported:
                assigned = "gpu"
            else:
                assigned = "cpu"
        else:
            assigned = "cpu"

        if current_unit is not None and current_unit != assigned:
            total_time += transfer_cost

        current_unit = assigned
        total_time += op_time.get(assigned, op_time.get("cpu", 1.0))

    return total_time


def evaluate_all_units(graph_profile):
    return {
        unit: compute_units_latency(graph_profile, unit)
        for unit in COMPUTE_UNITS
    }


def find_cpu_fallback_op(graph_profile):
    current_unit = None
    for op in graph_profile.get("nodes", []):
        supported = op.get("supported_units", ["cpu_only"])
        if "ane" in supported:
            assigned = "ane"
        elif "gpu" in supported:
            assigned = "gpu"
        else:
            assigned = "cpu"

        if (
            current_unit is not None
            and current_unit != "cpu"
            and assigned == "cpu"
        ):
            return op["name"]
        current_unit = assigned
    return None


ALLOWED_OPS = {"conv2d", "matmul", "depthwise_conv2d", "relu", "add"}


def is_ane_eligible(op_spec):
    op_type = op_spec.get("op_type")
    if op_type not in ALLOWED_OPS:
        return False

    dtype = op_spec.get("dtype")
    if dtype not in ("float16", "int8"):
        return False

    shape = op_spec.get("shape", [])
    if len(shape) != 4:
        return False

    batch, channels, height, width = shape
    if batch != 1:
        return False

    if height > 4096 or width > 4096:
        return False

    if op_type in ("conv2d", "depthwise_conv2d"):
        k_height = op_spec.get("kernel_height", 1)
        k_width = op_spec.get("kernel_width", 1)
        if k_height > 15 or k_width > 15:
            return False

    if op_type == "matmul":
        if channels % 16 != 0:
            return False

    return True
