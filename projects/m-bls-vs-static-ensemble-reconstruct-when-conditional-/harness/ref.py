import numpy as np

MOCK_REQUESTS = [
    {"id": i, "score": float(i % 10), "data": np.array([i, i + 1], dtype=np.float32)}
    for i in range(50)
]

def mock_feature_extractor(data):
    return np.mean(data)

def mock_heavy_model_a(data):
    return data * 2.0

def mock_heavy_model_b(data):
    return data + 10.0

def reference_bls_route(request, threshold=5.0):
    feat = mock_feature_extractor(request["data"])
    if feat >= threshold:
        res = mock_heavy_model_a(request["data"])
        branch = "model_a"
    else:
        res = mock_heavy_model_b(request["data"])
        branch = "model_b"
    return {"branch": branch, "result": res}

def reference_static_ensemble(request):
    feat = mock_feature_extractor(request["data"])
    res_a = mock_heavy_model_a(request["data"])
    res_b = mock_heavy_model_b(request["data"])
    return {"feat": feat, "res_a": res_a, "res_b": res_b}
