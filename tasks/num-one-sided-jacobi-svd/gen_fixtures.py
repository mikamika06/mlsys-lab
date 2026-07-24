"""Generate test fixtures for one-sided Jacobi SVD."""

import numpy as np
import json

matrices = [
    np.array([[2.0, 1.0, 0.0],
              [1.0, 3.0, 1.0],
              [0.0, 1.0, 2.0]], dtype=np.float64),
    np.array([[4.0, 1.0, 0.0, 0.5],
              [1.0, 3.0, 1.0, 0.0],
              [0.0, 1.0, 2.0, 1.0],
              [0.5, 0.0, 1.0, 5.0]], dtype=np.float64),
]

D1 = np.diag([10.0, 8.0, 5.0, 3.0, 1.0])
P1 = np.array([[0.6, -0.8, 0.0, 0.0, 0.0],
               [0.8,  0.6, 0.0, 0.0, 0.0],
               [0.0,  0.0, 1.0, 0.0, 0.0],
               [0.0,  0.0, 0.0, 1.0, 0.0],
               [0.0,  0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
matrices.append(P1 @ D1 @ P1.T)

D2 = np.diag([12.0, 9.0, 4.0, 1.0, 1.0])
P2 = np.array([[0.5, -0.5, -0.5, -0.5, 0.0],
               [0.5,  0.5, -0.5,  0.5, 0.0],
               [0.5,  0.5,  0.5, -0.5, 0.0],
               [0.5, -0.5,  0.5,  0.5, 0.0],
               [0.0,  0.0,  0.0,  0.0, 1.0]], dtype=np.float64)
matrices.append(P2 @ D2 @ P2.T)

rng = np.random.RandomState(42)
A_rand = rng.randn(6, 6)
matrices.append(A_rand @ A_rand.T + 0.1 * np.eye(6))

with open("fixtures.json", "w") as f:
    json.dump([m.tolist() for m in matrices])

print(f"Wrote {len(matrices)} test matrices to fixtures.json")
