import os
import tempfile
import ref

def check(workdir):
    from ollamautil.blobs import locate_and_verify_blob
    out = {"blob_found": 0.0, "hash_matched": 0.0}

    with tempfile.TemporaryDirectory() as tmp:
        dummy_content = b"GGUF_MOCK_WEIGHTS_DATA"
        dummy_path = os.path.join(tmp, "model.gguf")
        with open(dummy_path, "wb") as f:
            f.write(dummy_content)

        expected_hash = ref.compute_blob_hash(dummy_path)

        try:
            res = locate_and_verify_blob(dummy_path, expected_hash)
            if res and res.get("found"):
                out["blob_found"] = 1.0
                if res.get("hash") == expected_hash:
                    out["hash_matched"] = 1.0
        except Exception as e:
            out["_note"] = f"m2 failed: {type(e).__name__}: {str(e)[:100]}"
    return out
