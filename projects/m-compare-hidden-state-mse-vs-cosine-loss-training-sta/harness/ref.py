import numpy as np

np.random.seed(42)
STUDENT_STATES = np.random.randn(5, 16).astype(np.float32)
TEACHER_STATES = np.random.randn(10, 16).astype(np.float32)
