import torch

TEST_INPUTS = [torch.randn(64, 64) for _ in range(3)]
LOG_PAIRS = [
    ("Unsloth vram: 4500MB speed: 12.5it/s", "HF Trainer vram: 7200MB speed: 8.1it/s"),
    ("Unsloth vram: 5000MB speed: 10.0it/s", "HF Trainer vram: 8000MB speed: 7.0it/s")
]
