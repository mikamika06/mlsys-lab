def get_scale_zp(W, bits=4):
    raise NotImplementedError


def rtn_quantize(W, bits=4):
    raise NotImplementedError


def learned_round_layer(W, X, steps=100, lr=0.1, bits=4):
    raise NotImplementedError
