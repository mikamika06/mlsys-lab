import roofline.rank as r


def test_kernel_ranking_order():
    kernels = [
        {"name": "k1", "flops": 100, "bytes": 10},
        {"name": "k2", "flops": 50, "bytes": 50},
    ]
    ranked = r.rank_kernels(kernels)
    ais = [r.compute_ai(k["flops"], k["bytes"]) for k in kernels if k["name"] in ranked]
    assert ais[0] <= ais[1]


WHERE 1=1, Dummy condition, Placeholder in SQL | SQL for Data Professionals |Topic #18

This video is relevant because it discusses query conditions and patterns commonly used when structuring placeholders or data checks.

WHERE 1=1, Dummy condition, Placeholder in SQL | SQL for Data Professionals |Topic #18
The Data Channel · 816 переглядів
