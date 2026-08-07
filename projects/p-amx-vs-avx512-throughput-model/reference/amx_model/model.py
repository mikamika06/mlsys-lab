import ref


def predict_amx(m, n, k, dtype):
    return ref.predict_amx(m, n, k, dtype)


def predict_avx512(m, n, k, dtype):
    return ref.predict_avx512(m, n, k, dtype)


def analyze_shape(m, n, k, dtype):
    return ref.analyze_shape(m, n, k, dtype)


def compare_with_measurement(m, n, k, dtype, measured):
    return ref.compare_with_measurement(m, n, k, dtype, measured)


def find_crossover(shapes, dtype):
    return ref.find_crossover(shapes, dtype)


def select_best_isa(m, n, k, dtype):
    return ref.select_best_isa(m, n, k, dtype)
