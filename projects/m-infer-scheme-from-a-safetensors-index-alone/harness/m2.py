import ref

def check(workdir):
    from ctinspect.repair import repair_missing_weight_shape
    from ctinspect.calc import calculate_quant_byte_size

    out = {"repairs_matched": 0.0, "bytes_matched": 0.0}

    repair_ok = True
    for i, fixture in enumerate(ref.INDEX_FIXTURES):
        want_rep = ref.repair_missing_weight_shape(fixture)
        got_rep = repair_missing_weight_shape(fixture)
        if got_rep != want_rep:
            repair_ok = False
            out["_note"] = f"repair mismatch on fixture {i}: got {got_rep}, want {want_rep}"
            break

    if repair_ok:
        out["repairs_matched"] = 1.0

    bytes_ok = True
    test_cases = [
        ([128, 256], 4, True),
        ([128, 256], 4, False),
        ([512, 512], 2, True),
        ([512, 512], 8, False),
    ]
    for shape, bits, packed in test_cases:
        want_b = ref.calculate_quant_byte_size(shape, bits, packed)
        got_b = calculate_quant_byte_size(shape, bits, packed)
        if got_b != want_b:
            bytes_ok = False
            out["_note"] = f"calc mismatch for {shape}, bits={bits}, packed={packed}: got {got_b}, want {want_b}"
            break

    if bytes_ok:
        out["bytes_matched"] = 1.0

    return out
