def elementwise_kernel(n):
    out = bytearray()
    for i in range(n):
        out.extend(int(3 * i + 7).to_bytes(4, "little", signed=True))

    trace = []
    for i in range(n):
        trace.append(i * 4)
    for i in range(n):
        trace.append(4096 + i * 4)

    return bytes(out), trace
