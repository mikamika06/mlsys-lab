import sys
import os
import struct

sys.path.insert(0, ".")
from ggufwriter.streaming import GGUFTensorStreamer

def test_streamer_yields_items():
    path = "test_sample.gguf"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 2))
        f.write(struct.pack("<Q", 0))
    try:
        streamer = GGUFTensorStreamer(path)
        items = list(streamer)
        assert len(items) == 2, f"Expected 2 tensors, got {len(items)}"
    finally:
        if os.path.exists(path):
            os.remove(path)
