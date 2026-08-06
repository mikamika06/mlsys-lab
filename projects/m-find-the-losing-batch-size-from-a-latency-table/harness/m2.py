import ref

def check(workdir):
    from latency.tuning import tune_max_num_seqs
    latencies = [15.0, 25.0, 45.0, 90.0]
    seqs_list = [8, 16, 32, 64]
    slo = 50.0
    got = tune_max_num_seqs(latencies, slo, seqs_list)
    expected = 32
    match = 1.0 if got == expected else 0.0
    return {"ttft_tuned": match}
