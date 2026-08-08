import struct


def parse_signature_def(data):
    """Parse a custom binary signature def payload without external dependencies."""
    if not isinstance(data, bytes) or len(data) < 4:
        raise ValueError("Invalid data payload")
    num_inputs, num_outputs = struct.unpack(">HH", data[:4])
    offset = 4
    inputs = []
    for _ in range(num_inputs):
        name_len = struct.unpack(">H", data[offset:offset+2])[0]
        offset += 2
        name = data[offset:offset+name_len].decode("utf-8")
        offset += name_len
        dtype_len = struct.unpack(">H", data[offset:offset+2])[0]
        offset += 2
        dtype = data[offset:offset+dtype_len].decode("utf-8")
        offset += dtype_len
        ndims = struct.unpack(">B", data[offset:offset+1])[0]
        offset += 1
        shape = list(struct.unpack(">" + "I"*ndims, data[offset:offset+4*ndims])) if ndims > 0 else []
        offset += 4 * ndims
        inputs.append({"name": name, "dtype": dtype, "shape": shape})

    outputs = []
    for _ in range(num_outputs):
        name_len = struct.unpack(">H", data[offset:offset+2])[0]
        offset += 2
        name = data[offset:offset+name_len].decode("utf-8")
        offset += name_len
        dtype_len = struct.unpack(">H", data[offset:offset+2])[0]
        offset += 2
        dtype = data[offset:offset+dtype_len].decode("utf-8")
        offset += dtype_len
        ndims = struct.unpack(">B", data[offset:offset+1])[0]
        offset += 1
        shape = list(struct.unpack(">" + "I"*ndims, data[offset:offset+4*ndims])) if ndims > 0 else []
        offset += 4 * ndims
        outputs.append({"name": name, "dtype": dtype, "shape": shape})

    return {"inputs": inputs, "outputs": outputs}
