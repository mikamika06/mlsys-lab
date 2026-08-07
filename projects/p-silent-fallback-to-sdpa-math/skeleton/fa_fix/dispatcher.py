def get_backend(q, k, v, mask=None):
    raise NotImplementedError

def find_disqualification_reason(q, k, v, mask=None):
    raise NotImplementedError

def fix_inputs(q, k, v, mask=None):
    raise NotImplementedError

def measure_speedup(q, k, v, mask=None):
    raise NotImplementedError

def run_configs(configs):
    raise NotImplementedError

def strict_attention(q, k, v, mask=None, strict=True):
    raise NotImplementedError
