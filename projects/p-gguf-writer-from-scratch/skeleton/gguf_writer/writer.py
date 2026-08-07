import struct
import numpy as np

class GGUFWriter:
    def __init__(self, path):
        raise NotImplementedError

    def add_metadata(self, key, value):
        raise NotImplementedError

    def add_tensor(self, name, tensor):
        raise NotImplementedError

    def write(self):
        raise NotImplementedError
