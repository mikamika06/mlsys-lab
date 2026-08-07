import pickle


def load_checkpoint(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_checkpoint(state_dict, path):
    with open(path, "wb") as f:
        pickle.dump(state_dict, f)
