import ref

def check(workdir):
    from vllmlog.sharding import check_sharding
    ok = 0
    for item in ref.SHARDING_TESTS:
        want = ref.check_sharding(item["num_attention_heads"], item["num_kv_heads"], item["tensor_parallel_size"])
        got = check_sharding(item["num_attention_heads"], item["num_kv_heads"], item["tensor_parallel_size"])
        if got == want:
            ok += 1
    return {"sharding_match": 1.0 if ok == len(ref.SHARDING_TESTS) else 0.0}
