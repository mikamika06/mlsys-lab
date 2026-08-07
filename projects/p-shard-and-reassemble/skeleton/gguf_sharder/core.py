class GGUFSharder:
    def __init__(self, path):
        raise NotImplementedError

    def split(self, max_size, output_dir):
        raise NotImplementedError

    def verify_shards(self, output_dir):
        raise NotImplementedError

    def reassemble(self, output_dir, output_path):
        raise NotImplementedError

    def direct_tensor_load(self, output_dir, offset, length):
        raise NotImplementedError

    def compute_bpw(self):
        raise NotImplementedError
