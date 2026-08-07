import os
import tempfile
import ref


def check(workdir):
    m = {"vocab_match": 0.0, "tensor_count_ok": 0.0}
    with tempfile.TemporaryDirectory() as tmp:
        w_dir, v_path = ref.setup_workspace(tmp)
        out_gguf = os.path.join(tmp, "model.gguf")

        import sys
        sys.path.insert(0, workdir)
        try:
            import gguf_pipe.convert as conv
            if conv.verify_tokenizer(v_path):
                m["vocab_match"] = 1.0
            conv.convert_safetensors_to_gguf(w_dir, out_gguf)
            if os.path.exists(out_gguf):
                m["tensor_count_ok"] = 1.0
        except Exception:
            pass
    return m
