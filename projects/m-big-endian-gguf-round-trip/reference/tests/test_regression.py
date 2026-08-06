import numpy as np
import gguf_be.zero_copy as zc
from gguf_be.writer import write_gguf_be
from gguf_be.reader import read_gguf_be


def test_zero_copy_integrity():
    meta = {"model.name": "test_model", "general.alignment": 32}
    t_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    tensors = [{"name": "weight", "data": t_data, "dtype": "float32"}]

    buf = write_gguf_be(meta, tensors, alignment=32)
    read_meta, read_tensors, base_offset = read_gguf_be(buf)

    assert len(read_tensors) == 1
    t_info = read_tensors[0]

    extracted = zc.extract_tensor_zero_copy(buf, t_info, base_offset)

    assert extracted.shape == t_data.shape
    assert np.allclose(extracted, t_data)
    assert extracted.flags.owndata is False


def test_multiple_tensors_alignment():
    meta = {"general.alignment": 64}
    t1 = np.arange(10, dtype=np.int32)
    t2 = np.arange(20, dtype=np.float32)
    tensors = [
        {"name": "t1", "data": t1, "dtype": "int32"},
        {"name": "t2", "data": t2, "dtype": "float32"},
    ]

    buf = write_gguf_be(meta, tensors, alignment=64)
    _, read_tensors, base_offset = read_gguf_be(buf)

    for i, expected in enumerate([t1, t2]):
        info = read_tensors[i]
        ext = zc.extract_tensor_zero_copy(buf, info, base_offset)
        assert np.array_equal(ext, expected)
        assert ext.flags.owndata is False
