import ref


def check(workdir):
    from flopcount.attention import count_attention_flops

    max_rel_err = 0.0
    for case in ref.ATTN_TEST_CASES:
        want = ref.count_attention_flops(**case)
        got = count_attention_flops(**case)
        err = abs(float(got) - float(want)) / float(want)
        if err > max_rel_err:
            max_rel_err = err

    return {"rel_err": float(max_rel_err)}
