import json


def export_pte(model_dict):
    serialized = json.dumps(model_dict, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    return bytes(serialized, "utf-8")
