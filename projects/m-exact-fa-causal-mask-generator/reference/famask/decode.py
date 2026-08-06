import numpy as np
from famask.generator import generate_causal_mask


def decode_causal_mask(sk, alignment="top-left"):
    return generate_causal_mask(1, sk, alignment=alignment)
