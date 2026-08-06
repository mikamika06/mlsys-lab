import numpy as np

def generate_test_models():
    np.random.seed(1337)
    models = []
    shapes = [(64, 128), (128, 256), (256, 512)]
    for shape in shapes:
        model = {
            "proj_q": np.random.randn(*shape).astype(np.float32),
            "proj_k": np.random.randn(*shape).astype(np.float32),
            "proj_v": np.random.randn(*shape).astype(np.float32)
        }
        models.append(model)
    return models

TEST_MODELS = generate_test_models()
