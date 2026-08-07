import ref

def check(workdir):
    from exporter.optimizer import verify_output
    x, y = ref.get_sample_tensors()
    diff = verify_output(x, y)
    return {"max_diff": diff}
