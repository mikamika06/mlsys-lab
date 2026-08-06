import os
import tempfile
import numpy as np
from gguf_utils.writer import write_reference_gguf
from gguf_utils.patcher import patch_metadata_in_place
from gguf import GGUFReader


def test_metadata_patching_preserves_tensors():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.gguf")
        write_reference_gguf(path)

        reader_before = GGUFReader(path)
        tensor_before = reader_before.tensors[0].data.copy()

        patch_metadata_in_place(path, {"toy.block_count": 4})

        reader_after = GGUFReader(path)
        tensor_after = reader_after.tensors[0].data.copy()

        assert reader_after.fields["toy.block_count"].parts[reader_after.fields["toy.block_count"].data[0]] == 4
        np.testing.assert_array_equal(tensor_before, tensor_after)
