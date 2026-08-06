import relays.model as m
import relays.compare as c
import relays.fold as f

def get_reference_counts():
    model = m.make_3op_model()
    return c.compare_ir_counts(model)

def get_reference_folding():
    model = m.make_3op_model()
    return f.check_constant_folding(model)
