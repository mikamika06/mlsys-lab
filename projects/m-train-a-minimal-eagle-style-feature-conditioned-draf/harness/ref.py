import numpy as np
from typing import List, Dict

VOCAB_SIZE = 100
EMBED_DIM = 32
HIDDEN_DIM = 64
NUM_SAMPLES = 200


def get_synthetic_dataset(seed: int = 42):
    rng = np.random.RandomState(seed)
    token_ids = rng.randint(0, VOCAB_SIZE, size=(NUM_SAMPLES,))
    hidden_states = rng.randn(NUM_SAMPLES, HIDDEN_DIM)

    target_tokens = (hidden_states[:, 0] > 0).astype(int)
    return token_ids, hidden_states, target_tokens


def generate_vllm_eagle_logs(seed: int = 42) -> List[str]:
    rng = np.random.RandomState(seed)
    lines = []
    for i in range(50):
        acc = rng.randint(1, 6)
        lines.append(
            f"2026-08-07 10:00:00 [INFO] engine.py:120 -- EAGLE speculative batch execution step {i} - accepted_tokens: {acc}, draft_len: 5"
        )
    return lines


def ref_eagle_forward(e_head, token_ids: np.ndarray, hidden_states: np.ndarray) -> np.ndarray:
    t_emb = e_head.embed[token_ids]
    h_proj = np.dot(hidden_states, e_head.proj_feat)
    fused = np.concatenate([t_emb, h_proj], axis=-1)
    hidden = np.tanh(np.dot(fused, e_head.fc))
    return np.dot(hidden, e_head.head)


def ref_parse_vllm_eagle_log(log_lines: List[str]) -> Dict[str, float]:
    accs = []
    for line in log_lines:
        if "accepted_tokens:" in line:
            part = line.split("accepted_tokens:")[1].split(",")[0].strip()
            accs.append(int(part))
    return {
        "mean_accepted_length": float(np.mean(accs)) if accs else 0.0,
        "total_steps": float(len(accs)),
    }
