def evaluate_engine(int8_latency, fp16_latency, threshold=1.2):
    return "int8" if int8_latency * threshold < fp16_latency else "fp16"
