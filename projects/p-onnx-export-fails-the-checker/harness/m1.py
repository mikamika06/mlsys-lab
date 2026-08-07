import ref

def check(workdir):
    from exporter.fixer import classify_errors
    log = ref.get_sample_log()
    res = classify_errors(log)
    return {"errors_classified": float(len(res))}
