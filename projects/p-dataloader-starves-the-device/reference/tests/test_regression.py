from dl.config import predict_workers

def test_prediction_accuracy():
    res = predict_workers(100.0, 25.0)
    assert res == 4
    res2 = predict_workers(10.0, 100.0)
    assert res2 == 1
