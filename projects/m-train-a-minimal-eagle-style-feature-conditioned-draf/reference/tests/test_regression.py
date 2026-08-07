import sys
import numpy as np

sys.path.insert(0, ".")
from eagledraft.model import TokenOnlyDraftHead, EagleFeatureDraftHead
from eagledraft.eval import compare_draft_heads
from eagledraft.metrics import parse_vllm_eagle_log


def test_feature_head_outperforms_token_head():
    rng = np.random.RandomState(42)
    N, V, E, H = 200, 100, 32, 64
    token_ids = rng.randint(0, V, size=(N,))

    hidden_states = rng.randn(N, H)
    target_tokens = (hidden_states[:, 0] > 0).astype(int)

    t_head = TokenOnlyDraftHead(V, E, seed=42)
    e_head = EagleFeatureDraftHead(V, E, H, seed=42)

    e_head.proj_feat[0, :] = 5.0
    e_head.fc[:E, :] = 0.0
    e_head.fc[E:, :] = 5.0
    e_head.head[:, 0] = -10.0
    e_head.head[:, 1] = 10.0

    res = compare_draft_heads(t_head, e_head, token_ids, hidden_states, target_tokens)
    assert res["eagle_acc"] > res["token_acc"]


def test_parse_vllm_eagle_log_correctness():
    logs = [
        "2026-08-07 [INFO] EAGLE speculative step 1 - accepted_tokens: 3",
        "2026-08-07 [INFO] EAGLE speculative step 2 - accepted_tokens: 5",
        "2026-08-07 [INFO] EAGLE speculative step 3 - accepted_tokens: 1",
    ]
    res = parse_vllm_eagle_log(logs)
    assert abs(res["mean_accepted_length"] - 3.0) < 1e-5
    assert res["total_steps"] == 3.0
