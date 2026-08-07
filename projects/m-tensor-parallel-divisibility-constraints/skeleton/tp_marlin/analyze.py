MARLIN_IN_ALIGN = 128
MARLIN_OUT_ALIGN = 256
MARLIN_SPEED = 10000.0
FALLBACK_SPEED = 2500.0


def check_eligibility(layers, tp_size):
    raise NotImplementedError()


def evaluate_performance(layers, tp_size):
    raise NotImplementedError()


def pad_for_marlin(layers, tp_size):
    raise NotImplementedError()
