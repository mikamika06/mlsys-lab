import sys
import numpy as np

sys.path.insert(0, ".")
from ringattn.lse import merge_lse_pair, merge_partial_outputs


def test_lse_merge_accuracy():
    np.random.seed(42)
    s, d = 16, 32
    out1, max1, sum1 = np.random.randn(s, d), np.random.randn(s), np.exp(np.random.randn(s))
    out2, max2, sum2 = np.random.randn(s, d), np.random.randn(s), np.exp(np.random.randn(s))

    merged_out, merged_max, merged_sum = merge_lse_pair(out1, max1, sum1, out2, max2, sum2)

    for i in range(s):
        m1, m2 = max1[i], max2[i]
        l1, l2 = sum1[i], sum2[i]
        m_new = max(m1, m2)
        l_new = np.exp(m1 - m_new) * l1 + np.exp(m2 - m_new) * l2
        o_new = (np.exp(m1 - m_new) * l1 * out1[i] + np.exp(m2 - m_new) * l2 * out2[i]) / l_new

        np.testing.assert_allclose(merged_max[i], m_new, rtol=1e-5)
        np.testing.assert_allclose(merged_sum[i], l_new, rtol=1e-5)
        np.testing.assert_allclose(merged_out[i], o_new, rtol=1e-5)


def test_multi_partial_outputs():
    np.random.seed(101)
    s, d = 8, 16
    partials = []
    for _ in range(4):
        o = np.random.randn(s, d)
        m = np.random.randn(s)
        l = np.exp(np.random.randn(s))
        partials.append((o, m, l))

    res_out, res_max, res_sum = merge_partial_outputs(partials)
    assert res_out.shape == (s, d)
    assert res_max.shape == (s,)
    assert res_sum.shape == (s,)
