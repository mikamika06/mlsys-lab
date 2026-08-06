import ref


def check(workdir):
    from pipeline.prefetch import simulate_double_buffer
    data = ref.generate_test_data()
    want = ref.simulate_double_buffer(data["loads"], data["h2ds"], data["comps"])
    got = simulate_double_buffer(data["loads"], data["h2ds"], data["comps"])
    match = 1.0 if abs(got - want) < 1e-5 else 0.0
    return {"prefetch_match": match}
