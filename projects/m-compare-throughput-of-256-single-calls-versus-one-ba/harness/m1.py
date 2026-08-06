import ref


def check(workdir):
    from embedrun.runner import measure_throughput_ratio

    inputs = ref.get_test_inputs()
    ratio = measure_throughput_ratio(
        ref.mock_single_endpoint,
        ref.mock_batched_endpoint,
        inputs
    )
    out = {"throughput_ratio": float(ratio)}
    return out
