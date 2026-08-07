def lora_delta_forward(x: list[list[float]], base: list[list[float]], A: list[list[float]], B: list[list[float]], scale: float) -> list[list[float]]:
    n = len(x)
    d = len(x[0])
    r = len(A[0])

    xA = [[sum(x[i][k] * A[k][j] for k in range(d)) for j in range(r)] for i in range(n)]
    xA_B = [[sum(xA[i][k] * B[k][j] for k in range(r)) for j in range(d)] for i in range(n)]
    return [[base[i][j] + scale * xA_B[i][j] for j in range(d)] for i in range(n)]
