def static_buffer_replay(W: list[list[float]]):
    static_in = []
    static_out = []

    def replay(X: list[list[float]]) -> list[list[float]]:
        nonlocal static_in, static_out
        n = len(X)
        d = len(X[0]) if n > 0 else 0
        m = len(W)

        # Allocate/resize static input buffer
        if len(static_in) != n:
            static_in = [[0.0] * d for _ in range(n)]
        # Copy X into static input buffer
        for i in range(n):
            for j in range(d):
                static_in[i][j] = float(X[i][j])

        # Allocate/resize static output buffer
        if len(static_out) != n:
            static_out = [[0.0] * m for _ in range(n)]

        # Compute Y = static_in @ W.T into static output buffer
        for i in range(n):
            for k in range(m):
                s = 0.0
                for j in range(d):
                    s += static_in[i][j] * W[k][j]
                static_out[i][k] = s

        # Return a deep copy / independent snapshot of the static output buffer
        return [row[:] for row in static_out]

    return replay
