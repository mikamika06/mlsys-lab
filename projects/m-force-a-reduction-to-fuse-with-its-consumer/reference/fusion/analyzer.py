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
