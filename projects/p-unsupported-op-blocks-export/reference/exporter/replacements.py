import numpy as np

_catalog = {}

def substitute_op(tensor):
    return 0.5 * tensor * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (tensor + 0.044715 * np.power(tensor, 3))))

def verify_equivalence(original_fn, replaced_fn, test_data):
    orig = original_fn(test_data)
    rep = replaced_fn(test_data)
    diff = np.max(np.abs(orig - rep))
    return bool(diff < 1e-4)

def check_tolerance(output_orig, output_new, tol=1e-3):
    return bool(np.max(np.abs(output_orig - output_new)) <= tol)

def catalog_add(op_name, replacement_fn):
    _catalog[op_name] = replacement_fn
    return True
