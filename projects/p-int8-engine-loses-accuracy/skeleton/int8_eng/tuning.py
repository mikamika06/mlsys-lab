def find_sensitive_layers(profile_data, threshold=0.03):
    raise NotImplementedError

def relocate_qdq(graph, sensitive_layers):
    raise NotImplementedError

def calibrate(model, calibration_data):
    raise NotImplementedError

def evaluate_engine(model_int8, test_data):
    raise NotImplementedError
