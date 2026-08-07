import math


def _naive(z):
    ez_list = []
    s = 0.0
    for x in z:
        try:
            val = math.exp(x)
        except OverflowError:
            val = math.inf
        ez_list.append(val)
        s += val
    res = []
    for val in ez_list:
        if s == 0.0:
            res_val = math.nan if val == 0.0 else math.inf
        else:
            try:
                res_val = val / s
            except ZeroDivisionError:
                res_val = math.nan
        res.append(res_val)
    return res


def _lse(z):
    m = -math.inf
    for x in z:
        if x > m:
            m = x
    ez_list = []
    s = 0.0
    for x in z:
        try:
            val = math.exp(x - m)
        except OverflowError:
            val = math.inf
        ez_list.append(val)
        s += val
    res = []
    for val in ez_list:
        if s == 0.0:
            res_val = math.nan if val == 0.0 else math.inf
        else:
            try:
                res_val = val / s
            except ZeroDivisionError:
                res_val = math.nan
        res.append(res_val)
    return res


def _online(z):
    m = -math.inf
    s = 0.0
    for x in z:
        new_m = max(m, x)
        try:
            exp_diff_m = math.exp(m - new_m)
        except OverflowError:
            exp_diff_m = 0.0
        try:
            exp_diff_x = math.exp(x - new_m)
        except OverflowError:
            exp_diff_x = 0.0
        s = s * exp_diff_m + exp_diff_x
        m = new_m
    res = []
    for x in z:
        try:
            num = math.exp(x - m)
        except OverflowError:
            num = math.inf
        if s == 0.0:
            res_val = math.nan if num == 0.0 else math.inf
        else:
            try:
                res_val = num / s
            except ZeroDivisionError:
                res_val = math.nan
        res.append(res_val)
    return res


def _overflowed(p):
    for x in p:
        if not math.isfinite(x):
            return True
    return False


def classify_softmax_overflow(z: list[float]) -> tuple[bool, bool, bool]:
    """
    Run three softmax implementations on 1-D score vector z and report
    whether each one's output probability vector contains any inf/nan:
      1. naive: exp(z) / sum(exp(z)), no stabilization.
      2. lse: exp(z - max(z)) / sum(exp(z - max(z))).
      3. online: single-pass streaming softmax with a running max/sum,
         rescaling the running sum whenever the running max updates.

    Return (naive_overflow, lse_overflow, online_overflow).
    """
    return (_overflowed(_naive(z)), _overflowed(_lse(z)), _overflowed(_online(z)))
