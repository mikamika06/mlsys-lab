import ref


def check(workdir):
    from gradacc.memory import peak_memory_sweep

    W, b = ref.get_model()
    gen = ref.get_generator()
    steps_list = [1, 2, 4, 8]

    try:
        peaks = peak_memory_sweep(gen, W, b, steps_list)
    except Exception as e:
        return {"_note": f"crashed: {e}"}

    if not peaks or len(peaks) != 4:
        return {"_note": f"Expected 4 peaks, got {len(peaks) if peaks else 0}"}

    base_mem = peaks[0]
    max_mem = max(peaks)

    ratio = float(max_mem) / float(base_mem) if base_mem > 0 else 0.0

    return {"memory_ratio": ratio}
