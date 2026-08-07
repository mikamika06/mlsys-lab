import ref


def measure_latency(model_type):
    res = ref.measure_latency(model_type)
    return res["CPU_AND_NE"]
