import numpy as np
from cascade.sampling import cascade_stage1_accept, cascade_stage2_accept, multi_draft_select
from cascade.latency import cascade_latency_per_token, is_2stage_net_win, break_even_alpha2, expected_tokens


def test_cascade_stage1_accept_lossless_distribution():
    q1 = np.array([0.5, 0.3, 0.2])
    q2 = np.array([0.2, 0.5, 0.3])
    rng = np.random.default_rng(42)

    counts = np.zeros(3)
    num_samples = 20000
    for _ in range(num_samples):
        x1 = int(rng.choice(3, p=q1))
        acc, x2 = cascade_stage1_accept(q1, q2, x1, rng)
        counts[x2] += 1

    empirical = counts / num_samples
    assert np.max(np.abs(empirical - q2)) < 0.02


def test_cascade_stage2_accept_lossless_distribution():
    q2 = np.array([0.2, 0.5, 0.3])
    p = np.array([0.1, 0.2, 0.7])
    rng = np.random.default_rng(123)

    counts = np.zeros(3)
    num_samples = 20000
    for _ in range(num_samples):
        x2 = int(rng.choice(3, p=q2))
        acc, x_final = cascade_stage2_accept(q2, p, x2, rng)
        counts[x_final] += 1

    empirical = counts / num_samples
    assert np.max(np.abs(empirical - p)) < 0.02


def test_multi_draft_select_lossless():
    q0 = np.array([0.6, 0.3, 0.1])
    q1 = np.array([0.2, 0.6, 0.2])
    p = np.array([0.1, 0.1, 0.8])
    rng = np.random.default_rng(999)

    counts = np.zeros(3)
    num_samples = 20000
    for _ in range(num_samples):
        c0 = int(rng.choice(3, p=q0))
        c1 = int(rng.choice(3, p=q1))
        acc, idx, x_out = multi_draft_select([c0, c1], [q0, q1], p, rng)
        counts[x_out] += 1

    empirical = counts / num_samples
    assert np.max(np.abs(empirical - p)) < 0.02


def test_latency_decision_consistency():
    c1, g1 = 1.0, 5
    c2, g2 = 2.0, 3
    cT = 20.0
    alpha_direct = 0.6

    be = break_even_alpha2(c1, g1, c2, g2, cT, alpha_direct)

    win_below = is_2stage_net_win(c1, g1, c2, g2, cT, be - 0.05, alpha_direct)
    win_above = is_2stage_net_win(c1, g1, c2, g2, cT, be + 0.05, alpha_direct)

    assert not win_below
    assert win_above
