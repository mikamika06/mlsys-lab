import sys

sys.path.insert(0, ".")
from ctxshift import kv_cache_bytes, mha_vs_gqa, simulate

N_CTX, N_KEEP, N_TOKENS = 32, 5, 200


def test_kept_prefix_never_evicted():
    out = simulate(N_CTX, N_KEEP, N_TOKENS)
    leaked = set(range(N_KEEP)) & set(out["evicted"])
    assert not leaked, f"kept prefix ids leaked into evicted: {sorted(leaked)}"


def test_every_token_accounted_for_exactly_once():
    out = simulate(N_CTX, N_KEEP, N_TOKENS)
    seen = out["resident"] + out["evicted"]
    assert sorted(seen) == list(range(N_TOKENS)), "a token id was lost or duplicated"
    assert len(seen) == len(set(seen))


def test_resident_never_exceeds_window():
    out = simulate(N_CTX, N_KEEP, N_TOKENS)
    assert len(out["resident"]) <= N_CTX


def test_shift_events_never_negative():
    out = simulate(N_CTX, N_KEEP, N_TOKENS)
    assert all(e >= 0 for e in out["shift_events"])


def test_gqa_never_costs_more_than_mha():
    cfg = {"n_layers": 8, "n_heads": 8, "n_kv_heads": 2, "head_dim": 64,
           "n_ctx": 4096, "bytes_per_element": 2}
    cmp = mha_vs_gqa(cfg)
    assert cmp["gqa_bytes"] <= cmp["mha_bytes"]
    assert cmp["gqa_bytes"] == kv_cache_bytes(cfg)
