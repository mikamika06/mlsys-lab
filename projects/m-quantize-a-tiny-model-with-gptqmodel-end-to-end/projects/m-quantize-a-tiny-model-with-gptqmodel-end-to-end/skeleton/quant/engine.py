def get_calibration_data():
    raise NotImplementedError()


def build_calibration_dataset(model, inputs):
    raise NotImplementedError()


def quantize_weights(model, calib_data, bits=4):
    raise NotImplementedError()
