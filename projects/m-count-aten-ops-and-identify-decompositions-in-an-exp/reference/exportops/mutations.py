def capture_export_mutation_error(model_def, export_fn):
    try:
        export_fn(model_def)
        return None
    except Exception as e:
        return (type(e).__name__, str(e))
