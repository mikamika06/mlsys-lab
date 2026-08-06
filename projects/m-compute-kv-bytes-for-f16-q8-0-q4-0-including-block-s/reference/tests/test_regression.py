from kvquant.eval import eval_needle_retrieval


def test_needle_retrieval_quality():
    res_f16 = eval_needle_retrieval(dtype="f16")
    res_q8 = eval_needle_retrieval(dtype="q8_0")
    res_q4 = eval_needle_retrieval(dtype="q4_0")

    assert res_f16["needle_found"]
    assert res_q8["needle_found"]
    assert res_q4["needle_found"]

    assert res_f16["cosine_similarity"] > 0.999
    assert res_q8["cosine_similarity"] > 0.99
    assert res_q4["cosine_similarity"] > 0.95

    assert res_q4["rel_l2_error"] > res_q8["rel_l2_error"]
