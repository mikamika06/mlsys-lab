def verify_copy_cost(original_manifest, copied_manifest):
    orig_digests = {l["digest"] for l in original_manifest.get("layers", [])}
    copy_digests = {l["digest"] for l in copied_manifest.get("layers", [])}
    if orig_digests != copy_digests:
        return False
    return 0
