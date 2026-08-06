import numpy as np

def compare_sizes(rows: int, cols: int, bits_a: int, bits_b: int) -> dict:
    size_a = (rows * cols * bits_a + 7) // 8
    size_b = (rows * cols * bits_b + 7) // 8
    return {"size_a": int(size_a), "size_b": int(size_b), "ratio": float(size_a / size_b)}
