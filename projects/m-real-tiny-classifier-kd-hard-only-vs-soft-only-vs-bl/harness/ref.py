import numpy as np

np.random.seed(42)
SAMPLE_LOGITS = np.random.randn(10, 5)
SAMPLE_TARGETS = np.random.randint(0, 5, size=10)
SAMPLE_TEACHER_LOGITS = np.random.randn(10, 5)
