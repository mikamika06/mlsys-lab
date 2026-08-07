import os

def create_dummy_model(workdir):
    os.makedirs(os.path.join(workdir, "edge"), exist_ok=True)
    os.makedirs(os.path.join(workdir, "tests"), exist_ok=True)
    path = os.path.join(workdir, "model.tflite")
    with open(path, "w") as f:
        f.write("dummy_tflite_content")
    return path
