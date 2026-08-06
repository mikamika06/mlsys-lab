import numpy as np

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_I32 = 2
GGML_TYPE_I16 = 3
GGML_TYPE_I8 = 4

GGML_TO_BE_DTYPE = {
    GGML_TYPE_F32: np.dtype(">f4"),
    GGML_TYPE_F16: np.dtype(">f2"),
    GGML_TYPE_I32: np.dtype(">i4"),
    GGML_TYPE_I16: np.dtype(">i2"),
    GGML_TYPE_I8: np.dtype("i1"),
}


def extract_tensor_zero_copy(buffer, tensor_info, data_base_offset):
    """Extract a zero-copy numpy array view from buffer for a tensor info dict."""
    abs_offset = data_base_offset + tensor_info["offset"]
    shape = tuple(tensor_info["shape"])
    ggml_type = tensor_info["ggml_type"]
    be_dtype = GGML_TO_BE_DTYPE[ggml_type]
    count = int(np.prod(shape))
    view = np.frombuffer(buffer, dtype=be_dtype, count=count, offset=abs_offset)
    return view.reshape(shape)
