import sys


def check(workdir):
    out = {"blobs_valid": 0.0, "models_created": 0.0}
    sys.path.insert(0, workdir)
    try:
        from runner.remote_blobs import MockBlobServer, push_safetensors_as_model
    except Exception as e:
        out["_note"] = f"Failed to import runner.remote_blobs: {e}"
        return out

    server = MockBlobServer()
    b1 = b"weights_chunk_1_content"
    b2 = b"weights_chunk_2_content"

    res1 = server.upload_blob(b1)
    res2 = server.upload_blob(b2)

    if res1.get("status") == "success" and "digest" in res1 and res1.get("size") == len(b1):
        out["blobs_valid"] = 1.0
    else:
        out["_note"] = f"upload_blob failed: {res1}"
        return out

    files_map = {"model-00001.safetensors": b1, "model-00002.safetensors": b2}
    create_res = push_safetensors_as_model(server, "test-model:latest", files_map)
    model_data = server.get_model("test-model:latest")

    if (
        create_res.get("status") == "created"
        and model_data is not None
        and model_data.get("total_size") == len(b1) + len(b2)
        and len(model_data.get("digests", [])) == 2
    ):
        out["models_created"] = 1.0
    else:
        out["_note"] = f"push_safetensors_as_model failed: {create_res}, {model_data}"

    return out
