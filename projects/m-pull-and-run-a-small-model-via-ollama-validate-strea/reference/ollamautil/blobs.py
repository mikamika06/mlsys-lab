import os
import hashlib

def locate_and_verify_blob(filepath, expected_hash):
    if not os.path.isfile(filepath):
        return {"found": False, "hash": ""}
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            block = f.read(8192)
            if not block:
                break
            sha.update(block)
    file_hash = sha.hexdigest()
    matched = (file_hash == expected_hash)
    return {"found": True, "hash": file_hash, "matched": matched}
