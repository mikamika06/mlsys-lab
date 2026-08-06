WIDTH = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
         "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
         "U32": 4, "I32": 4, "F32": 4,
         "U64": 8, "I64": 8, "F64": 8}


class SafetensorsError(Exception):
    pass


def parse_header(blob):
    """{header_bytes, data_start, header, metadata}.

    The file opens with the header length as a little-endian unsigned 64-bit
    integer, then that many bytes of JSON. Tensor data begins immediately
    after. `__metadata__` is a reserved key and not a tensor.
    """
    raise NotImplementedError


def entries(blob):
    """{data_start, metadata, tensors}, tensors ordered by where they sit.

    Per tensor: name, dtype, shape, elements, width, declared_bytes,
    expected_bytes, relative_offsets, absolute_offsets. The offsets in the
    header are relative to the start of the data section, and the difference
    between what a tensor declares and what its dtype and shape need is the
    whole point of the next function.
    """
    raise NotImplementedError


def validate(blob):
    """Structural problems, as a list of strings, empty when the file is sound.

    Catch: a dtype nobody knows, a declared length that its shape does not
    account for, a gap or overlap between consecutive tensors, and data that
    ends past the end of the file. Each message names the tensor.
    """
    raise NotImplementedError
