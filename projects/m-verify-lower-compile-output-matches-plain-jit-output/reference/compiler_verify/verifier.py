import numpy as np


def verify_compilation(x, expected_out, compiled_out, max_abs_err=1e-5):
    diff = np.max(np.abs(compiled_out - expected_out))
    return float(diff) <= max_abs_err


def inspect_stablehlo(text):
    ops = {}
    for line in text.splitlines():
        if "stablehlo." in line:
            parts = line.strip().split()
            for p in parts:
                if p.startswith("stablehlo."):
                    op_name = p.split("(")[0]
                    ops[op_name] = ops.get(op_name, 0) + 1
    unique_ops = sorted(list(ops.keys()))
    return ops, unique_ops


def verify_export_roundtrip(x, expected_out, exported_out, max_abs_err=1e-5):
    diff = np.max(np.abs(exported_out - expected_out))
    return float(diff) <= max_abs_err
