import ref

def check(workdir):
    from attencurve.analyzer import find_crossover_length
    lengths, cust, base = ref.get_test_data()
    want = ref.find_crossover_length(lengths, cust, base)
    got = find_crossover_length(lengths, cust, base)
    out = {"crossover_match": 1.0 if got == want else 0.0}
    if got != want:
        out["_note"] = f"got crossover length {got}, reference {want}"
    return out
