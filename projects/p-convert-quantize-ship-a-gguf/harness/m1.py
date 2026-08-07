import os
import struct
import sys
import tempfile
import ref

def check(workdir):
    out = {"gguf_header_valid": 0.0, "tensor_count_match": 0.0, "vocab_integrity": 0.0}

    sys.path.insert(0, workdir)
    from gguf_pipeline.converter import GGUFConverter

    hf_tensors = ref.create_dummy_hf_model()
    vocab = ref.get_dummy_vocab()

    converter = GGUFConverter(model_config={"hidden_size": 64}, vocab=vocab)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "model.gguf")
        converter.export_gguf(hf_tensors, out_path)

        if not os.path.exists(out_path):
            return out

        with open(out_path, "rb") as f:
            magic = f.read(4)
            if magic == b"GGUF":
                out["gguf_header_valid"] = 1.0

            ver, n_tensors, vocab_size = struct.unpack("<IQQ", f.read(20))
            if n_tensors == len(hf_tensors):
                out["tensor_count_match"] = 1.0

            if vocab_size == len(vocab):
                out["vocab_integrity"] = 1.0

    return out
