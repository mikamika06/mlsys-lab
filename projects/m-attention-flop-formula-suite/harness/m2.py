import ref


def check(workdir):
    from flopcount.varlen import count_varlen_attention_flops, flops_from_histogram

    max_rel_err = 0.0
    for case in ref.VARLEN_TEST_CASES:
        if "seq_lens" in case:
            want = ref.count_varlen_attention_flops(**case)
            got = count_varlen_attention_flops(**case)
        else:
            want = ref.flops_from_histogram(**case)
            got = flops_from_histogram(**case)

        err = abs(float(got) - float(want)) / float(want)
        if err > max_rel_err:
            max_rel_err = err

    return {"rel_err": float(max_rel_err)}
