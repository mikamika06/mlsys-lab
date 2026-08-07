class QuantConfig:
    def __init__(self, bits=4, group_size=128, sym=True, damp_percent=0.01, desc_act=False):
        self.bits = bits
        self.group_size = group_size
        self.sym = sym
        self.damp_percent = damp_percent
        self.desc_act = desc_act

    def to_dict(self):
        return {
            "bits": self.bits,
            "group_size": self.group_size,
            "sym": self.sym,
            "damp_percent": self.damp_percent,
            "desc_act": self.desc_act
        }

def make_config(bits=4, group_size=128, sym=True, damp_percent=0.01, desc_act=False):
    return QuantConfig(bits=bits, group_size=group_size, sym=sym, damp_percent=damp_percent, desc_act=desc_act)
