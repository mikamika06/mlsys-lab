import json


def load_graph(data):
    if isinstance(data, str):
        return json.loads(data)
    elif isinstance(data, bytes):
        return json.loads(data.decode("utf-8"))
    elif isinstance(data, dict):
        return data
    raise TypeError("Unsupported data type for graph loading")
