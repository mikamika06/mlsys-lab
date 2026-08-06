import struct

MAGIC = b"GGUF"

UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64 = range(13)

TYPE_NAME = {
    UINT8: "UINT8", INT8: "INT8", UINT16: "UINT16", INT16: "INT16",
    UINT32: "UINT32", INT32: "INT32", FLOAT32: "FLOAT32", BOOL: "BOOL",
    STRING: "STRING", ARRAY: "ARRAY", UINT64: "UINT64", INT64: "INT64",
    FLOAT64: "FLOAT64",
}

# ggml type id -> (elements per block, bytes per block). Fill in the ones you need.
BLOCK = {
    0: (1, 4),
    12: (256, 144),
    14: (256, 210),
}

GGML_NAME = {0: "F32", 1: "F16", 12: "Q4_K", 14: "Q6_K"}


class GGUFError(Exception):
    pass


def parse_header(blob):
    """{magic, version, tensor_count, kv_count, cursor}.

    cursor is the byte offset where the key/value section begins. Raise
    GGUFError on a bad magic or a version this reader does not implement.
    """
    raise NotImplementedError


def parse_kv(blob):
    """{"kv": {key: value}, "types": {key: type name}, "cursor": offset}.

    Every GGUF value type has to round-trip, arrays included. An array's type
    name is reported as ARRAY[ELEMENT], for example ARRAY[STRING].
    """
    raise NotImplementedError


def parse_tensor_index(blob):
    """{"alignment": int, "data_start": int, "tensors": [...]}.

    Each tensor is {name, shape_ggml_order, ggml_type_id, ggml_type,
    relative_offset, n_elements, n_bytes, absolute_data_offset}. The data
    section starts at the first general.alignment boundary at or after the end
    of the tensor index, and every relative_offset is measured from there.
    """
    raise NotImplementedError


def validate(blob):
    """Every structural problem with this container, as a list of strings.

    An empty list means the file is safe to memory-map. A tensor whose data
    runs past the end of the file, an offset that is not aligned, two tensors
    whose data overlap: each is one entry naming the tensor it is about.
    """
    raise NotImplementedError


def tensor_bytes(blob, info):
    """The raw bytes of one tensor, given its entry from parse_tensor_index."""
    raise NotImplementedError
