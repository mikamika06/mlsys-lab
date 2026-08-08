import sys

sys.path.insert(0, ".")
from ggftool.writer import GGUFWriter
from ggftool.dump import dump_json
from ggftool.patch import patch_metadata_in_place

def test_metadata_and_tensor_alignment() -> None:
    writer = GGUFWriter(alignment=64)
    writer.add_uint32("general.version", 1)
    writer.add_string("general.name", "test-model-v1")
    writer.add_float32("general.temperature", 0.7)

    payload_a = bytes([i % 256 for i in range(128)])
    payload_b = bytes([(i * 3) % 256 for i in range(256)])

    writer.add_tensor("blk.0.weight", [16, 8], 0, payload_a)
    writer.add_tensor("blk.1.weight", [32, 8], 0, payload_b)

    gguf_bin = writer.write()
    dumped = dump_json(gguf_bin)

    assert dumped["alignment"] == 64
    assert dumped["metadata"]["general.name"] == "test-model-v1"
    assert len(dumped["tensors"]) == 2

    patched = patch_metadata_in_place(gguf_bin, {"general.name": "test-model-v2"})
    dumped_patched = dump_json(patched)

    assert dumped_patched["metadata"]["general.name"] == "test-model-v2"

    orig_dump = dump_json(gguf_bin)
    data_base = gguf_bin.find(payload_a[:16])
    assert data_base != -1
    assert gguf_bin[data_base:] == patched[data_base:]
