def functional_add(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Return a new array equal to a + b."""
    return [[a[i][j] + b[i][j] for j in range(len(a[i]))] for i in range(len(a))]

def functional_relu(x: list[list[float]]) -> list[list[float]]:
    """Return a new array with ReLU applied elementwise."""
    return [[x[i][j] if x[i][j] > 0 else 0.0 for j in range(len(x[i]))] for i in range(len(x))]

def functional_copy(a: list[list[float]]) -> list[list[float]]:
    """Return a copy of the input array."""
    return [[a[i][j] for j in range(len(a[i]))] for i in range(len(a))]
