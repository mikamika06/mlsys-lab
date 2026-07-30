import sys

sys.path.insert(0, ".")
import numpy as np

from gqa.mapping import build_head_map, build_query_groups
from gqa.attention import expand_kv, gqa_attention

NUM_Q_HEADS = 8
NUM_KV_HEADS = 2
GROUP = NUM_Q_HEADS // NUM_KV_HEADS


def test_head_map_is_contiguous_blocks():
    head_map = build_head_map(NUM_Q_HEADS, NUM_KV_HEADS)
    assert len(head_map) == NUM_Q_HEADS
    assert all(0 <= k < NUM_KV_HEADS for k in head_map)
    assert all(head_map[i] <= head_map[i + 1] for i in range(len(head_map) - 1)), \
        f"head map is not block-ordered: {head_map}"
    for k in range(NUM_KV_HEADS):
        members = [q for q in range(NUM_Q_HEADS) if head_map[q] == k]
        expected = list(range(k * GROUP, (k + 1) * GROUP))
        assert members == expected, f"kv head {k} owns {members}, expected contiguous {expected}"


def test_query_groups_match_head_map():
    head_map = build_head_map(NUM_Q_HEADS, NUM_KV_HEADS)
    groups = build_query_groups(NUM_Q_HEADS, NUM_KV_HEADS)
    assert len(groups) == NUM_KV_HEADS
    seen = sorted(q for g in groups for q in g)
    assert seen == list(range(NUM_Q_HEADS)), "every query head must land in exactly one group"
    for k, g in enumerate(groups):
        assert sorted(g) == [q for q in range(NUM_Q_HEADS) if head_map[q] == k]


def test_expand_kv_pulls_from_the_right_block():
    kv = np.arange(NUM_KV_HEADS, dtype=np.float64).reshape(NUM_KV_HEADS, 1, 1)
    expanded = expand_kv(kv, NUM_Q_HEADS)
    assert expanded.shape == (NUM_Q_HEADS, 1, 1)
    for q in range(NUM_Q_HEADS):
        assert expanded[q, 0, 0] == q // GROUP, f"query head {q} pulled from the wrong kv head"


def test_gqa_attention_runs_and_is_causal():
    rng = np.random.default_rng(0)
    seq, head_dim = 5, 4
    q = rng.standard_normal((NUM_Q_HEADS, seq, head_dim))
    k = rng.standard_normal((NUM_KV_HEADS, seq, head_dim))
    v = rng.standard_normal((NUM_KV_HEADS, seq, head_dim))
    out = gqa_attention(q, k, v, NUM_KV_HEADS)
    assert out.shape == (NUM_Q_HEADS, seq, head_dim)
    assert np.all(np.isfinite(out))
    k2, v2 = k.copy(), v.copy()
    k2[:, -1, :] += 50.0
    v2[:, -1, :] += 50.0
    out2 = gqa_attention(q, k2, v2, NUM_KV_HEADS)
    assert np.allclose(out[:, :-1, :], out2[:, :-1, :]), \
        "changing the last key/value leaked into earlier positions"
