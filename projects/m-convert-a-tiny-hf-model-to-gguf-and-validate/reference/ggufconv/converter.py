_REGISTRY = {}

def register_converter(name, cls):
    _REGISTRY[name] = cls

class SynthModelConverter:
    def __init__(self, config):
        self.config = config

    def convert_tensors(self, tensors):
        out = {}
        for k, v in tensors.items():
            new_k = k.replace("base_model.", "")
            out[new_k] = v
        return out

register_converter("synth_tiny", SynthModelConverter)
