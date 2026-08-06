PLANS = [
    {
        "operations": [
            {"id": "conv_0", "type": "conv2d", "dispatched_device": "ANE", "estimated_cost": 12.5, "ane_supported": True, "ane_rejection_reason": None},
            {"id": "relu_0", "type": "relu", "dispatched_device": "ANE", "estimated_cost": 1.0, "ane_supported": True, "ane_rejection_reason": None},
            {"id": "soft_0", "type": "softmax", "dispatched_device": "CPU", "estimated_cost": 4.2, "ane_supported": False, "ane_rejection_reason": "UNSUPPORTED_OPERATOR"}
        ]
    },
    {
        "operations": [
            {"id": "attn_qkv", "type": "matmul", "dispatched_device": "ANE", "estimated_cost": 45.0, "ane_supported": True, "ane_rejection_reason": None},
            {"id": "ln_0", "type": "layer_norm", "dispatched_device": "GPU", "estimated_cost": 8.0, "ane_supported": False, "ane_rejection_reason": "PRECISION_FALLBACK"},
            {"id": "reshape_0", "type": "reshape", "dispatched_device": "CPU", "estimated_cost": 2.1, "ane_supported": False, "ane_rejection_reason": "DYNAMIC_SHAPE"}
        ]
    },
    {
        "operations": [
            {"id": "c1", "type": "conv2d", "dispatched_device": "ANE", "estimated_cost": 10.0, "ane_supported": True, "ane_rejection_reason": None},
            {"id": "c2", "type": "conv2d", "dispatched_device": "ANE", "estimated_cost": 15.0, "ane_supported": True, "ane_rejection_reason": None},
            {"id": "p1", "type": "max_pool", "dispatched_device": "ANE", "estimated_cost": 3.0, "ane_supported": True, "ane_rejection_reason": None},
            {"id": "d1", "type": "dense", "dispatched_device": "ANE", "estimated_cost": 20.0, "ane_supported": True, "ane_rejection_reason": None}
        ]
    },
    {
        "operations": [
            {"id": "cust_1", "type": "fft", "dispatched_device": "CPU", "estimated_cost": 30.0, "ane_supported": False, "ane_rejection_reason": "UNSUPPORTED_OPERATOR"},
            {"id": "cust_2", "type": "custom_arg_max", "dispatched_device": "GPU", "estimated_cost": 18.0, "ane_supported": False, "ane_rejection_reason": "COST_EXCEEDED"}
        ]
    },
    {
        "operations": [
            {"id": "embed", "type": "embedding", "dispatched_device": "CPU", "estimated_cost": 50.0, "ane_supported": False, "ane_rejection_reason": "DYNAMIC_SHAPE"},
            {"id": "gelu", "type": "gelu", "dispatched_device": "ANE", "estimated_cost": 5.0, "ane_supported": True, "ane_rejection_reason": None},
            {"id": "proj", "type": "linear", "dispatched_device": "ANE", "estimated_cost": 25.0, "ane_supported": True, "ane_rejection_reason": None},
            {"id": "norm", "type": "rms_norm", "dispatched_device": "GPU", "estimated_cost": 7.0, "ane_supported": False, "ane_rejection_reason": "PRECISION_FALLBACK"}
        ]
    }
]


def parse_compute_plan(plan_dict):
    """Parse raw compute plan into structured list of op records."""
    ops = []
    raw_ops = plan_dict.get("operations", [])
    for op in raw_ops:
        dev = op.get("dispatched_device", "CPU")
        reason = op.get("ane_rejection_reason")
        if dev == "ANE":
            reason = None
        ops.append({
            "id": str(op.get("id", "")),
            "type": str(op.get("type", "")),
            "device": str(dev),
            "cost": float(op.get("estimated_cost", 0.0)),
            "ane_supported": bool(op.get("ane_supported", False)),
            "ane_rejection_reason": reason,
        })
    return ops


def routing_fractions(parsed_ops):
    """Compute device routing fractions across ANE, GPU, and CPU."""
    if not parsed_ops:
        return {"ANE": 0.0, "GPU": 0.0, "CPU": 0.0}
    counts = {"ANE": 0, "GPU": 0, "CPU": 0}
    for op in parsed_ops:
        dev = op.get("device", "CPU")
        if dev in counts:
            counts[dev] += 1
        else:
            counts["CPU"] += 1
    total = len(parsed_ops)
    return {dev: counts[dev] / total for dev in ["ANE", "GPU", "CPU"]}


def find_ane_rejections(parsed_ops):
    """Find ops rejected from ANE and return their details and cost information."""
    rejections = []
    for op in parsed_ops:
        if op.get("device") != "ANE" and op.get("ane_rejection_reason"):
            rejections.append({
                "id": op["id"],
                "type": op["type"],
                "device": op["device"],
                "cost": op["cost"],
                "reason": op["ane_rejection_reason"],
            })
    return rejections
