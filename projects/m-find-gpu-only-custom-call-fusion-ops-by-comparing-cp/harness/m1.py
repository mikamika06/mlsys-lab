import ref


def check(workdir):
    from hlodiff.parser import parse_hlo_ops
    cpu_text, gpu_text = ref.generate_sample_dumps()
    want = ref.parse_hlo(gpu_text)
    try:
        got = parse_hlo_ops(gpu_text)
    except Exception as e:
        return {"parsed_correctly": 0.0, "_note": f"raised {e}"}
    if sorted(got) == sorted(want):
        return {"parsed_correctly": 1.0}
    return {"parsed_correctly": 0.0, "_note": f"got {got}, want {want}"}
