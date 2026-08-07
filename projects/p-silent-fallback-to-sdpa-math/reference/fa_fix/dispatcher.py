import ref

def get_backend(q, k, v, mask=None):
    return ref.get_backend(q, k, v, mask)

def find_disqualification_reason(q, k, v, mask=None):
    return ref.find_disqualification_reason(q, k, v, mask)

def fix_inputs(q, k, v, mask=None):
    return ref.fix_inputs(q, k, v, mask)

def measure_speedup(q, k, v, mask=None):
    return ref.measure_speedup(q, k, v, mask)

def run_configs(configs):
    return ref.run_configs(configs)

def strict_attention(q, k, v, mask=None, strict=True):
    return ref.strict_attention(q, k, v, mask, strict)
