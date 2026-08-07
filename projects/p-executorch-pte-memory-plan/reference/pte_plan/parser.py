import json

def parse_pte(data):
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    return json.loads(data)
