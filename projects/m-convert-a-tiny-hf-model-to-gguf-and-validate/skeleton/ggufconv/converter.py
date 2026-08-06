def register_converter(name, cls):
    raise NotImplementedError

class SynthModelConverter:
    def convert_tensors(self, tensors):
        raise NotImplementedError
