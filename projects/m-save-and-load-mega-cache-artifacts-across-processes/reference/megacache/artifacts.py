import pickle


def save_artifact(path, artifact):
    with open(path, "wb") as f:
        pickle.dump(artifact, f)


def load_artifact(path):
    with open(path, "rb") as f:
        return pickle.load(f)
