import numpy as np
from gguf import GGUFWriter


def write_reference_gguf(path: str) -> None:
    writer = GGUFWriter(path, arch="toy")
    writer.add_uint32("toy.block_count", 2)
    writer.add_float32("toy.attention.layer_norm_rms_epsilon", 1e-5)
    writer.add_string("toy.description", "reference toy model")

    tensor_data = np.arange(16, dtype=np.float32).reshape(4, 4)
    writer.add_tensor("toy.weight", tensor_data)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
