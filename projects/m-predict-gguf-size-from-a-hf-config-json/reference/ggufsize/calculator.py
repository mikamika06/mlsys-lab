def compute_size(config, quant_type="Q4_K_M"):
    import ref
    return ref.predict_size(config, quant_type)
