def capture_mutation_error(export_func, sample_inputs):
    try:
        export_func(*sample_inputs)
        return None
    except Exception as e:
        return type(e).__name__
