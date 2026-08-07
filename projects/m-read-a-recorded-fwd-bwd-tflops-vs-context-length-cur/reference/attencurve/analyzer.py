def find_crossover_length(lengths, custom_tflops, baseline_tflops):
    for l, c, b in zip(lengths, custom_tflops, baseline_tflops):
        if c >= b:
            return int(l)
    return int(lengths[-1])
