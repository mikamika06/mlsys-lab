def check(workdir):
    import ref
    from mlx_serve.quantize import quantize_model
    res = quantize_model(bits=4, max_gb=36.0)
    expected = ref.simulate_quantization(4, 36.0)
    ok = 1.0 if res.get("valid") == expected["valid"] else 0.0
    return {"quantize_ok": ok}
