import dis


def classify_thread_scaling(workloads):
    result = {}

    for item in workloads:
        fn = item["fn"]
        instructions = list(dis.get_instructions(fn))
        names = set(fn.__code__.co_names)

        has_python_loop = any(
            ins.opname in {"FOR_ITER", "JUMP_BACKWARD"}
            for ins in instructions
        )
        has_python_arithmetic = any(
            ins.opname == "BINARY_OP"
            for ins in instructions
        )

        scales = False

        if has_python_loop and has_python_arithmetic:
            scales = False
        elif "sleep" in names:
            scales = True
        elif "compress" in names:
            scales = True
        elif "dot" in names:
            scales = True

        result[item["name"]] = scales

    return result
