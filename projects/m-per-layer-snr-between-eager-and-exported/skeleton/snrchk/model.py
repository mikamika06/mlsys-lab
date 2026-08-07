class BundledProgram:
    def __init__(self, weights, diverge_layer=-1):
        raise NotImplementedError

    def run_exported(self, x):
        raise NotImplementedError

def run_eager(weights, x):
    raise NotImplementedError
