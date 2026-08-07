def substitute_op(tensor):
    raise NotImplementedError

def verify_equivalence(original_fn, replaced_fn, test_data):
    raise NotImplementedError

def check_tolerance(output_orig, output_new, tol):
    raise NotImplementedError

def catalog_add(op_name, replacement_fn):
    raise NotImplementedError
