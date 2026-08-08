import hashlib


def upload_and_create_model(blob_bytes, model_name):
    sha = hashlib.sha256(blob_bytes).hexdigest()
    return {"status": "success", "digest": f"sha256:{sha}", "model": model_name}
