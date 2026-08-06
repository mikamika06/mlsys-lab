"""Reference generator and fixture generator for harness checks."""
import random


def generate_vram_cases():
    random.seed(42)
    cases = []
    for _ in range(10):
        vanilla = round(random.uniform(8.0, 80.0), 2)
        pct = round(random.uniform(20.0, 80.0), 1)
        expected = round(vanilla * (1.0 - pct / 100.0), 4)
        cases.append({"vanilla": vanilla, "pct": pct, "expected": expected})
    return cases


def generate_log_cases():
    return [
        {
            "log": "Unsloth: Peak memory reserved: 18.50 GB\n{'loss': 2.31}\n{'loss': 1.42}\n5.0 steps/s",
            "want": {"peak_vram_gb": 18.5, "steps_per_sec": 5.0, "final_loss": 1.42},
            "vanilla_sps": 2.0,
            "want_speedup": 2.5,
        },
        {
            "log": "Peak VRAM allocated: 24.0 GB\nStep 10 loss = 0.987\n0.25 s/step",
            "want": {"peak_vram_gb": 24.0, "steps_per_sec": 4.0, "final_loss": 0.987},
            "vanilla_sps": 1.6,
            "want_speedup": 2.5,
        },
    ]
