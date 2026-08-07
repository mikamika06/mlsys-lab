import numpy as np

class Pruner:
    def __init__(self, model, groups):
        self.model = model
        self.groups = groups

    def prune_group(self, group, indices):
        for name in group:
            if name in self.model:
                w = self.model[name]
                if w.ndim == 2:
                    if name.endswith("in"):
                        self.model[name] = np.delete(w, indices, axis=1)
                    else:
                        self.model[name] = np.delete(w, indices, axis=0)
