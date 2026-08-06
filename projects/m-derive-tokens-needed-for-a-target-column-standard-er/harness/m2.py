import os
import ref


def check(workdir):
    from imatrix.convert import convert_dat_to_gguf
    dat_path = ref.generate_dat_fixture()
    gguf_path = os.path.join(workdir, "test_output.gguf")
    ok = 0
    try:
        convert_dat_to_gguf(dat_path, gguf_path)
        if os.path.exists(gguf_path):
            with open(gguf_path, "rb") as f:
                content = f.read()
            if content.startswith(b"GGUF") and b"IMATRIX_DUMMY_PAYLOAD_BYTES" in content:
                ok = 1
    except Exception:
        ok = 0
    finally:
        if os.path.exists(dat_path):
            os.remove(dat_path)
        if os.path.exists(gguf_path):
            os.remove(gguf_path)
    return {"conversion_match": float(ok)}
