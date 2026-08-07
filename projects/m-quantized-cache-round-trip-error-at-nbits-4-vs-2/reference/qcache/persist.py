import pickle

def save_cache(cache_data, path):
    with open(path, "wb") as f:
        pickle.dump(cache_data, f)

def load_cache(path):
    with open(path, "rb") as f:
        return pickle.load(f)
