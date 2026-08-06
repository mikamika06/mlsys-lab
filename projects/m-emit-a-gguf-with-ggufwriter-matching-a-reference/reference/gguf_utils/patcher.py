import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType


def patch_metadata_in_place(path: str, new_kv: dict) -> None:
    reader = GGUFReader(path)
    tensors = []
    for tensor in reader.tensors:
        tensors.append((tensor.name, tensor.data.copy(), tensor.tensor_type))

    arch = "toy"
    for k, v in reader.fields.items():
        if k == "general.architecture":
            arch = str(v.parts[v.data[0]])

    writer = GGUFWriter(path, arch=arch)
    for k, v in reader.fields.items():
        if k in ("general.architecture", "general.alignment"):
            continue
        if k in new_kv:
            val = new_kv[k]
            if isinstance(val, int):
                writer.add_uint32(k, val)
            elif isinstance(val, float):
                writer.add_float32(k, val)
            elif isinstance(val, str):
                writer.add_string(k, val)
        else:
            if v.type == GGUFValueType.UINT32:
                writer.add_uint32(k, int(v.parts[v.data[0]]))
            elif v.type == GGUFValueType.FLOAT32:
                writer.add_float32(k, float(v.parts[v.data[0]]))
            elif v.type == GGUFValueType.STRING:
                s_val = str(v.parts[v.data[0]])
                writer.add_string(k, s_val)

    for name, data, t_type in tensors:
        writer.add_tensor(name, data, tensor_type=t_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
