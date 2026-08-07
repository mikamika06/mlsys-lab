import json
import struct


def export_to_safetensors(state_dict, output_path):
    header = {}
    data_sections = []
    current_offset = 0

    for name, tensor in state_dict.items():
        tensor_bytes = tensor.tobytes()
        length = len(tensor_bytes)
        shape = list(tensor.shape)
        dtype = "F32"

        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [current_offset, current_offset + length]
        }
        data_sections.append(tensor_bytes)
        current_offset += length

    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_length = len(header_json)

    with open(output_path, "wb") as f:
        f.write(struct.pack("<Q", header_length))
        f.write(header_json)
        for section in data_sections:
            f.write(section)
    return output_path
