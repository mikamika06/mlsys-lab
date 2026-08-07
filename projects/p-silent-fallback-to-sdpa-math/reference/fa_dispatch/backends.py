import ref

def get_backend_name(q, k, v, mask):
    return ref.simulate_backend_choice(q, k, v, mask)

def dispatch_check(q, k, v, mask):
    backend = get_backend_name(q, k, v, mask)
    valid = ref.validate_constraints(q, k, v, mask, backend)
    return valid, backend
