from long_ctx.analyzer import isolate_failure_mode

def check(workdir):
    m = {"tokenization_isolated": 0.0}
    try:
        res1 = isolate_failure_mode({"middle_attention": 0.05}, {"unk_rate": 0.0})
        res2 = isolate_failure_mode({"middle_attention": 0.8}, {"unk_rate": 0.1})
        if res1 == "attention_failure" and res2 == "tokenization_failure":
            m["tokenization_isolated"] = 1.0
    except Exception:
        pass
    return m
