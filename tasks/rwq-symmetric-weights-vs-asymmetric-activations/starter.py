def ort_default_scheme(tags):
    """
    Given a list of tensor role tags ("weight" or "activation"), return a
    list of (qmin, qmax, is_symmetric) triples matching ONNX Runtime's
    default static-quantization scheme for that role. See task.md.
    """
    raise NotImplementedError('your code here')
