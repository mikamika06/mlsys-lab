def observe_ranges(activations):
    raise NotImplementedError


def compute_qparams(tensor, per_channel=False, axis=0):
    raise NotImplementedError


def convert_tensor(tensor, qparams):
    raise NotImplementedError
