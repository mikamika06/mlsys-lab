class QuantConfig:
    def __init__(self, bits=4, group_size=128, sym=True, damp_percent=0.01, desc_act=False):
        raise NotImplementedError

    def to_dict(self):
        raise NotImplementedError

def make_config(bits=4, group_size=128, sym=True, damp_percent=0.01, desc_act=False):
    raise NotImplementedError
