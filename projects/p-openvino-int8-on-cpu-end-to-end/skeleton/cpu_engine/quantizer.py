class Quantizer:
    def __init__(self, quant_bits=8):
        raise NotImplementedError

    def calibrate(self, graph, calibration_dataset):
        raise NotImplementedError

    def quantize(self, graph, calibration_dataset):
        raise NotImplementedError

    def compute_accuracy_loss(self, fp32_output, int8_output):
        raise NotImplementedError
