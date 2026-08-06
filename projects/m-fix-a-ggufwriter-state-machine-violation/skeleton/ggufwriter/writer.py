class GGUFWriter:
    def __init__(self, path, endian="little"):
        raise NotImplementedError

    def add_header(self, kv):
        raise NotImplementedError

    def add_tensor(self, name, data, tensor_type=None):
        raise NotImplementedError

    def write_header_to_file(self):
        raise NotImplementedError
