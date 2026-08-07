def predict_amx(m, n, k, dtype):
    raise NotImplementedError


def predict_avx512(m, n, k, dtype):
    raise NotImplementedError


def analyze_shape(m, n, k, dtype):
    raise NotImplementedError


def compare_with_measurement(m, n, k, dtype, measured):
    raise NotImplementedError


def find_crossover(shapes, dtype):
    raise NotImplementedError


def select_best_isa(m, n, k, dtype):
    raise NotImplementedError
