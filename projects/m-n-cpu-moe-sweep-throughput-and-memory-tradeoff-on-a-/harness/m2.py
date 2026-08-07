import ref


def check(workdir):
    from moe_sweep.verify import verify_placement
    tensor_map, overrides = ref.get_placement_data()
    want = ref.verify_placement(tensor_map, overrides)
    got = verify_placement(tensor_map, overrides)
    match = 1.0 if got == want else 0.0
    return {"placement_match": match}
