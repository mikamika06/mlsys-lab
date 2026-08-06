import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from ddpplan.bucket import build_bucket_plan

    configs = ref.generate_param_configs(seed=42)
    matched = True
    for params, cap in configs:
        want = ref.build_bucket_plan(params, cap)
        got = build_bucket_plan(params, cap)
        if got != want:
            matched = False
            break

    out = {"bucket_plans_matched": 1.0 if matched else 0.0}
    return out
