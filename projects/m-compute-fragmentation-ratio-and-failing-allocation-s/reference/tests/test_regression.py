import oomdiag.trend as t


def test_trend_prediction():
    steps = list(range(1, 10))
    mems = [100 + i * 50 for i in steps]
    cap = 1000
    res = t.predict_oom_step(steps, mems, cap)
    assert isinstance(res, int)
    assert res > 0
