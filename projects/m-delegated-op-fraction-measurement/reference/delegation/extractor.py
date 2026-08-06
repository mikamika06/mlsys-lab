def extract_delegate_info(model):
    blobs = []
    for node in model.get("nodes", []):
        if node.get("delegate_blob"):
            blobs.append({
                "node": node["name"],
                "backend": node["backend"],
                "blob": node["delegate_blob"]
            })
    return blobs
