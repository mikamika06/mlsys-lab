import ref


def check(workdir):
    from fp8att.hadamard import apply_hadamard_transform
    q, k, v = ref.generate_test_inputs()
    transformed = apply_hadamard_transform(q)
    score = ref.measure_incoherence(transformed)
    ref_score = ref.measure_incoherence(ref.apply_hadamard(q))
    val = 1.0 if score <= ref_score * 1.1 else 0.5
    if score < 5.0:
        val = 1.0
    return {"incoherence_score": float(val)}
