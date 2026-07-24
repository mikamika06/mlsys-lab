def prove_memory_sharing(values, index, new_value):
    buf = bytearray(values)
    first = memoryview(buf)
    second = memoryview(buf)

    before = second[index]
    first[index] = new_value

    return before, first[index], second[index]
