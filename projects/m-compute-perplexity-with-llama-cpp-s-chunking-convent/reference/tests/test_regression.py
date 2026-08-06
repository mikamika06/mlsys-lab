import os
import tempfile
import numpy as np
from ppl.chunking import compute_perplexity
from ppl.metrics import compute_logit_metrics
from ppl.quant_eval import dump_f16_logits, score_quantized_model


def test_perplexity_basic():
    tokens = [0, 1, 2, 3, 4, 5, 6, 7]

    def model_fn(chunk):
        L = len(chunk)
        res = np.zeros((L, 10))
        for i, tok in enumerate(chunk):
            res[i, (tok + 1) % 10] = 5.0
        return res

    ppl = compute_perplexity(model_fn, tokens, chunk_size=3)
    assert ppl > 0.0
    assert np.isfinite(ppl)


def test_logit_metrics():
    base = np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0]])
    quant = np.array([[2.0, 0.5, 0.0], [0.1, 2.9, 1.0]])
    res = compute_logit_metrics(base, quant)
    assert res["mean_kld"] >= 0.0
    assert res["top1_agreement"] == 1.0


def test_quant_eval_pipeline():
    tokens = [0, 1, 2, 3, 4, 5]
    vocab_size = 8

    def model_f16(chunk):
        L = len(chunk)
        res = np.zeros((L, vocab_size))
        for i in range(L):
            res[i, i % vocab_size] = 4.0
        return res

    def model_quant(chunk):
        L = len(chunk)
        res = np.zeros((L, vocab_size))
        for i in range(L):
            res[i, (i + 2) % vocab_size] = 4.0
        return res

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "f16.npy")
        dump_f16_logits(model_f16, tokens, chunk_size=2, output_path=path)
        scores = score_quantized_model(model_quant, tokens, chunk_size=2, f16_logits_path=path)
        assert scores["top1_agreement"] == 0.0
        assert scores["mean_kld"] > 0.0
