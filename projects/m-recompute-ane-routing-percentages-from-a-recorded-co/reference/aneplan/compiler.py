import os


def verify_mlmodelc(bundle_path):
    if not os.path.isdir(bundle_path):
        return False
    metadata_path = os.path.join(bundle_path, "model.espresso.net")
    weights_path = os.path.join(bundle_path, "model.espresso.weights")
    return os.path.isfile(metadata_path) and os.path.isfile(weights_path)
