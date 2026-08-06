"""Reference data generators and mock implementations for harness execution."""

import time

class MockTrainer:
    def __init__(self, peak_bytes):
        self._peak_bytes = peak_bytes

    def get_peak_memory_bytes(self):
        return self._peak_bytes

def create_mock_step_fns(pt_delay=0.002, mlx_delay=0.001):
    def pt_step():
        time.sleep(pt_delay)

    def mlx_step():
        time.sleep(mlx_delay)

    return pt_step, mlx_step

def get_sample_loss_curves():
    pt_losses = [3.0 - 0.2 * i for i in range(10)]
    mlx_losses = [3.0 - 0.21 * i for i in range(10)]
    return pt_losses, mlx_losses
