def generate_sample_dumps():
    cpu_hlo = """
    HloModule module_foo
    ENTRY %main (p0: f32[4,8]) -> f32[4,8] {
      %p0 = f32[4,8] parameter(0)
      ROOT %abs = f32[4,8] abs(f32[4,8] %p0)
    }
    """
    gpu_hlo = """
    HloModule module_foo
    ENTRY %main (p0: f32[4,8]) -> f32[4,8] {
      %p0 = f32[4,8] parameter(0)
      %fusion = f32[4,8] fusion(f32[4,8] %p0), kind=kInput, calls=%fusion_computation
      ROOT %custom = f32[4,8] custom-call(f32[4,8] %fusion), custom_call_target="MyGpuOp"
    }
    """
    return cpu_hlo, gpu_hlo

def parse_hlo(text):
    ops = []
    for line in text.splitlines():
        line = line.strip()
        if "custom-call" in line or "fusion" in line:
            ops.append(line)
    return ops

def find_gpu_only_ops(cpu_text, gpu_text):
    cpu_ops = set(parse_hlo(cpu_text))
    gpu_ops = set(parse_hlo(gpu_text))
    return sorted(list(gpu_ops - cpu_ops))

def simulate_compile(hlo_text):
    if "mismatch" in hlo_text or "invalid" in hlo_text:
        raise RuntimeError("XLA compile error: shape mismatch in custom-call")
    return True
