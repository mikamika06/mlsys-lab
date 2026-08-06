import ref

def check(workdir):
    from tflite_tools.quant import find_degenerate_quantization_params

    cases = ref.get_test_cases()
    matched = 1
    for case in cases:
        fb_bytes = ref.generate_mock_flatbuffer(case["tensors"])
        got = find_degenerate_quantization_params(fb_bytes)
        want = case["expected_degenerate"]
        if sorted(list(got)) != sorted(list(want)):
            matched = 0
            break

    return {"degenerate_match": float(matched)}
