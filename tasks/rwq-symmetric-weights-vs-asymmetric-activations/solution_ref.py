def ort_default_scheme(tags):
    """
    ONNX Runtime's default static-quantization scheme: weights are QInt8
    (signed, symmetric, zero_point forced to 0, codes in [-127, 127] to
    stay perfectly symmetric); activations are QUInt8 (unsigned,
    asymmetric, zero_point calibrated from data, codes in [0, 255]).
    Returns a list of (qmin, qmax, is_symmetric) triples, one per tag.
    """
    out = []
    for tag in tags:
        if tag == "weight":
            out.append((-127, 127, True))
        elif tag == "activation":
            out.append((0, 255, False))
        else:
            raise ValueError(f"unknown tensor tag: {tag!r}")
    return out
