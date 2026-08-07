import struct
import numpy as np

MAGIC = b"GGUF"
VERSION = 3

class GGUFConverter:
    """Converts HuggingFace tensor formats and vocabularies into GGUF model files."""

    def __init__(self, model_config, vocab):
        self.config = model_config
        self.vocab = vocab

    def convert_tensors(self, hf_tensors):
        converted = {}
        for name, tensor in hf_tensors.items():
            new_name = name
            if name.startswith("model."):
                new_name = name[6:]
            converted[new_name] = np.asarray(tensor, dtype=np.float32)
        return converted

    def export_gguf(self, hf_tensors, output_path):
        tensors = self.convert_tensors(hf_tensors)
        with open(output_path, "wb") as f:
            f.write(MAGIC)
            f.write(struct.pack("<I", VERSION))
            f.write(struct.pack("<Q", len(tensors)))
            f.write(struct.pack("<Q", len(self.vocab)))

            for token in self.vocab:
                encoded = token.encode("utf-8")
                f.write(struct.pack("<I", len(encoded)))
                f.write(encoded)

            for name, arr in tensors.items():
                encoded_name = name.encode("utf-8")
                f.write(struct.pack("<I", len(encoded_name)))
                f.write(encoded_name)
                f.write(struct.pack("<I", len(arr.shape)))
                for dim in arr.shape:
                    f.write(struct.pack("<Q", dim))
                data = arr.tobytes()
                f.write(struct.pack("<Q", len(data)))
                f.write(data)
