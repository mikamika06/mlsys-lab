import ref


def check(workdir):
    from fp8att.analysis import compute_attention_error
    q, k, v = ref.generate_test_inputs()
    want = ref.simulate_bf16_attention(q, k, v)
    got = compute_attention_error(q, k, v)
    err = ref.compute_relative_error(got, want)
    return {"rel_err": float(err)}
