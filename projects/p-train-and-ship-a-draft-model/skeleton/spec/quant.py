def quantize_weights(w):
    raise NotImplementedError

def dequantize_weights(w_q, w_min, scale):
    raise NotImplementedError

class QuantizedDraft:
    def __init__(self, draft):
        raise NotImplementedError

    def get_probs(self, token: int):
        raise NotImplementedError
