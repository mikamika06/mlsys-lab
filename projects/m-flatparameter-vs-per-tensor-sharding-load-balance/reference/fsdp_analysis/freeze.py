def check_freeze_constraint(parameters):
    for p in parameters:
        if p.get("requires_grad", True) and p.get("is_frozen", False):
            return "freeze_constraint_violation"
    return "valid"
