KERNELS_DUMP = """
@triton.jit
def triton_per_fused_add_mul_0(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel
    rgen = tl.arange(0, RBLOCK)
    tmp0 = tl.load(in_ptr0 + xindex)
    tmp1 = tl.sum(tmp0, 0)
    tl.store(out_ptr0 + xindex, tmp1)

@triton.jit
def triton_poi_fused_relu_1(in_ptr0, out_ptr0, xnumel):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    tmp0 = tl.load(in_ptr0 + xindex)
    tmp1 = tl.maximum(tmp0, 0.0)
    tl.store(out_ptr0 + xindex, tmp1)
"""

CPP_DUMP = """
void kernel_cpu(const float* in, float* out, long n) {
    #pragma omp parallel for
    for (long i = 0; i < n; i += 8) {
        __m256 x = _mm256_loadu_ps(in + i);
        __m256 y = _mm256_add_ps(x, _mm256_set1_ps(1.0f));
        _mm256_storeu_ps(out + i, y);
    }
    for (long j = 0; j < n; ++j) {
        out[j] = out[j] * 2.0f;
    }
}
"""

def classify_kernels(dump_text):
    results = {}
    for line in dump_text.splitlines():
        if "def triton_" in line:
            name = line.split("def ")[1].split("(")[0]
            if "_per_" in name or "_red_" in name:
                results[name] = "reduction"
            elif "_poi_" in name:
                results[name] = "pointwise"
            else:
                results[name] = "unknown"
    return results

def count_vectorized_loops(cpp_text):
    count = 0
    for line in cpp_text.splitlines():
        if "_mm256_" in line or "_mm512_" in line or "#pragma omp" in line:
            count += 1
    return count

def check_fusion_validity(analysis_result):
    return analysis_result.get("fused", False)
