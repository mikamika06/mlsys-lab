def generate_log(fps: float, median: float) -> str:
    return f"""[Step 11/11] Dumping statistics report
[ INFO ] Count:            100 iterations
[ INFO ] Duration:         1000.00 ms
[ INFO ] Latency:
[ INFO ]    Median:        {median:.2f} ms
[ INFO ]    Average:       {median + 0.1:.2f} ms
[ INFO ]    Min:           {median - 1.0:.2f} ms
[ INFO ]    Max:           {median + 2.0:.2f} ms
[ INFO ] Throughput:       {fps:.2f} FPS
"""

def generate_pc_log(layers: list) -> str:
    lines = [
        "[ INFO ] layerName                                          execStatus    layerType       execType                 realTime (ms)  cpuTime (ms)",
        "[ INFO ] -------------------------------------------------------------------------------------------------------------------------------------"
    ]
    for name, rtime in layers:
        lines.append(f"[ INFO ] {name:<50} EXECUTED      Convolution     jit_avx512               {rtime:.3f}          {rtime:.3f}")
    lines.append("[ INFO ] Total time: 100.000     milliseconds")
    return "\n".join(lines)
    
def get_cases_m1():
    cases = [
        (generate_log(123.45, 8.12), {"fps": 123.45, "median": 8.12}),
        (generate_log(50.00, 20.00), {"fps": 50.00, "median": 20.00}),
        (generate_log(8.91, 110.55), {"fps": 8.91, "median": 110.55}),
    ]
    return cases
    
def get_cases_m2():
    layers1 = [("conv1", 1.2), ("conv2", 5.5), ("fc", 10.1), ("pool", 0.5), ("relu", 0.1), ("conv3", 3.3)]
    want1 = [("fc", 10.1), ("conv2", 5.5), ("conv3", 3.3), ("conv1", 1.2), ("pool", 0.5)]
    
    layers2 = [("L"+str(i), float(i)) for i in range(10)]
    want2 = [("L9", 9.0), ("L8", 8.0), ("L7", 7.0), ("L6", 6.0), ("L5", 5.0)]
    
    return [
        (generate_pc_log(layers1), want1),
        (generate_pc_log(layers2), want2)
    ]
