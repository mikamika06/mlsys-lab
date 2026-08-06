import ref


def check(workdir):
    from realign.metrics import compute_byte_exact_fraction

    fractions = []
    for case in ref.CASES:
        f = compute_byte_exact_fraction(case["bytes_orig"], case["bytes_real"])
        fractions.append(f)
    avg_f = sum(fractions) / len(fractions) if fractions else 0.0
    return {"byte_exact_fraction": float(avg_f)}
