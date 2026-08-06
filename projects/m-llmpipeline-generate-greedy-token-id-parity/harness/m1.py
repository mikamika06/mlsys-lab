import sys
import time
import ref

def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    import inference

    model = ref.MockModel()
    pipeline = ref.MockPipeline(model)
    prompt_ids = [42, 100, 256]
    max_tokens = 20

    out = {"parity_match": 0.0, "throughput_ratio": 0.0}

    t0 = time.time()
    got_pipe = inference.run_pipeline(pipeline, prompt_ids, max_tokens)
    t1 = time.time()
    pipe_dur = max(t1 - t0, 0.0001)

    t2 = time.time()
    got_hand = inference.run_hand_rolled(model, prompt_ids, max_tokens)
    t3 = time.time()
    hand_dur = max(t3 - t2, 0.0001)

    expected = ref.MockPipeline(model).generate(prompt_ids, max_tokens)
    if list(got_pipe) == list(got_hand) and list(got_pipe) == list(expected):
        out["parity_match"] = 1.0
    else:
        out["_note"] = f"mismatch. pipe={got_pipe[:3]}, hand={got_hand[:3]}, exp={expected[:3]}"

    out["throughput_ratio"] = hand_dur / pipe_dur

    return out
