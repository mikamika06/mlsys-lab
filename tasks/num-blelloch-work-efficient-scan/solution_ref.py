def blelloch_scan(values):
    a = list(values)
    n = len(a)

    stride = 1
    while stride < n:
        step = stride * 2
        for i in range(0, n, step):
            a[i + step - 1] += a[i + stride - 1]
        stride = step

    a[-1] = 0

    stride = n // 2
    while stride >= 1:
        step = stride * 2
        for i in range(0, n, step):
            left = i + stride - 1
            right = i + step - 1
            t = a[left]
            a[left] = a[right]
            a[right] += t
        stride //= 2

    return a
