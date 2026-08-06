import math

def classify_extension(inv_freq: list[float]) -> str:
    n = len(inv_freq)
    if n < 2:
        return "None"

    std = []
    for i in range(n):
        std.append(1.0 / (10000.0 ** (2.0 * i / n)))

    k_sum = 0.0
    for i in range(n):
        k_sum += float(inv_freq[i]) / std[i]
    k_mean = k_sum / n

    if abs(k_mean - 1.0) > 1e-6:
        return "PI"

    r_std_expected = std[1] / std[0]
    r_ntk_expected = math.sqrt(r_std_expected)
    atol = 1e-6 * abs(r_ntk_expected)
    rtol = 1e-5

    all_close = True
    for i in range(n - 1):
        ratio = float(inv_freq[i + 1]) / float(inv_freq[i])
        if abs(ratio - r_ntk_expected) > (atol + rtol * abs(r_ntk_expected)):
            all_close = False
            break

    if all_close:
        return "NTK"

    diffs = []
    for i in range(n - 1):
        diffs.append(float(inv_freq[i + 1]) - float(inv_freq[i]))

    m = len(diffs)
    diffs_sum = 0.0
    abs_diffs_sum = 0.0
    for i in range(m):
        diffs_sum += diffs[i]
        abs_diffs_sum += abs(diffs[i])

    diffs_mean = diffs_sum / m
    mean_abs_diffs = abs_diffs_sum / m

    var_sum = 0.0
    for i in range(m):
        var_sum += (diffs[i] - diffs_mean) ** 2
    diffs_std = math.sqrt(var_sum / m)

    if diffs_std < 1e-6 * mean_abs_diffs:
        return "YaRN"

    return "None"
