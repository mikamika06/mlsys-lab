def _reference(size_histogram, bucket_size):
    total = 0
    for b, count in size_histogram.items():
        waste = (bucket_size - (b % bucket_size)) % bucket_size
        total += count * waste
    return int(total)

def grade(sol, fx) -> dict:
    cases = [
        ({3: 2, 5: 1}, 4),          # mixed sizes
        ({8: 3, 10: 2}, 5),         # includes a multiple of bucket
        ({4: 1}, 4),                # exact multiple only
        ({7: 0, 9: 5}, 3),          # zero count case
        ({12: 10, 15: 20, 18: 30}, 6)  # larger numbers
    ]
    ok = 1.0
    for hist, bucket in cases:
        try:
            got = sol.total_padding_waste(hist, bucket)
        except Exception:
            return {"exact_match": 0.0}
        ref = _reference(hist, bucket)
        if got != ref:
            return {"exact_match": 0.0}
    return {"exact_match": ok}
