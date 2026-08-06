import numpy as np

def mock_feature_extractor(data):
    return np.mean(data)

def mock_heavy_model_a(data):
    return data * 2.0

def mock_heavy_model_b(data):
    return data + 10.0

def route_request(request, threshold=5.0):
    data = request["data"]
    feat = mock_feature_extractor(data)
    if feat >= threshold:
        res = mock_heavy_model_a(data)
        branch = "model_a"
    else:
        res = mock_heavy_model_b(data)
        branch = "model_b"
    return {"branch": branch, "result": res}
