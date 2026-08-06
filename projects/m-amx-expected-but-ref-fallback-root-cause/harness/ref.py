import random

def generate_verbose_logs(seed=42):
    random.seed(seed)
    primitives = ["convolution", "matmul", "reorder", "eltwise", "pooling"]
    impls_good = ["jit:avx512_core", "jit:amx_bf16", "jit:avx2", "brgconv:avx512"]
    impls_ref = ["ref:any", "reference:gemm", "ref:matmul"]

    logs = []
    # Generate 10 exec logs
    for i in range(10):
        kind = random.choice(primitives)
        if i in [2, 5, 8]:
            impl = random.choice(impls_ref)
            shape = "mb1ic32ih224oc64_unaligned"
            aux = "aux:data_type_mismatch"
        else:
            impl = random.choice(impls_good)
            shape = "mb1ic32ih224oc64"
            aux = "aux:ok"
        t = round(random.uniform(0.1, 15.0), 3)
        log = f"onednn,exec,cpu,{kind},{impl},entry,{shape},{aux},{t}"
        logs.append(log)
    return logs

def generate_k_sweep_data(seed=42):
    random.seed(seed)
    records = []
    k_values = [16, 32, 64, 128, 256, 512, 1024, 2048]
    for k in k_values:
        if k < 64:
            isa = "avx2"
            lat = round(random.uniform(5.0, 8.0), 3)
        elif k < 256:
            isa = "avx512_core"
            lat = round(random.uniform(2.5, 4.0), 3)
        else:
            isa = "avx512_core_amx"
            lat = round(random.uniform(0.5, 1.2), 3)
        records.append({'k': k, 'isa': isa, 'latency_ms': lat})
    return records
