import numpy as np
import ref


def check(workdir):
    from cudagraphs.capture import GraphCaptureSimulator

    cases = ref.generate_trace_cases()
    matched = 0.0

    for case in cases:
        ref_res = ref.run_capture_ref(case["trace"], case["warmup"], case["replay"])

        sim = GraphCaptureSimulator(case["trace"])
        sim.warmup(case["warmup"])
        sim.capture(case["warmup"])
        got_res = sim.replay(case["replay"])

        equal = True
        for k in ref_res:
            if k not in got_res or not np.allclose(ref_res[k], got_res[k], atol=1e-6):
                equal = False
                break
        if equal:
            matched += 1.0

    return {"traces_matched": matched}
