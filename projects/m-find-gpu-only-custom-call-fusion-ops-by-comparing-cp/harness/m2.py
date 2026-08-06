import ref


def check(workdir):
    from hlodiff.diff import find_gpu_only_ops
    cpu_text, gpu_text = ref.generate_sample_dumps()
    want = ref.find_gpu_only_ops(cpu_text, gpu_text)
    try:
        got = find_gpu_only_ops(cpu_text, gpu_text)
    except Exception as e:
        return {"fusions_matched": 0.0, "_note": f"raised {e}"}
    if sorted(got) == sorted(want):
        return {"fusions_matched": 1.0}
    return {"fusions_matched": 0.0, "_note": f"got {got}, want {want}"}
