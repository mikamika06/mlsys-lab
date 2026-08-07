import numpy as np

def get_recorded_losses(workers: int) -> list[float]:
    np.random.seed(42)
    losses = []
    loss = 10.0
    for i in range(100):
        if workers >= 64 and i == 45:
            loss = 5000.0
        else:
            loss = loss * 0.95 + float(np.random.rand()) * 0.1
        losses.append(loss)
    return losses

def buggy_reduce(tensors: list[np.ndarray]) -> np.ndarray:
    res = np.zeros_like(tensors[0], dtype=np.float16)
    for t in tensors:
        res += t.astype(np.float16)
    return res.astype(np.float32)

def get_test_tensors() -> list[np.ndarray]:
    np.random.seed(1337)
    tensors = [np.array([10000.0, 5.0], dtype=np.float32)]
    for _ in range(30):
        tensors.append(np.array([1.0, 2.0], dtype=np.float32))
    return tensors
