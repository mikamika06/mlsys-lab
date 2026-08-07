import json

def parse_pte(data):
    if isinstance(data, str):
        return json.loads(data)
    return data
