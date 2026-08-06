from benchrunner.portability import diagnose_portability_failure


def test_missing_cpu_flags_rejected():
    host = {
        "arch": "x86_64",
        "cpu_flags": ["AVX"],
        "page_size_kb": 4,
        "kernel_version": "5.15.0",
    }
    binary = {
        "supported_arches": ["x86_64"],
        "required_cpu_flags": ["AVX", "AVX512F"],
        "supported_page_sizes_kb": [4],
        "min_kernel_version": "5.0.0",
    }
    result = diagnose_portability_failure(host, binary)
    assert not result["is_compatible"]
    assert result["status"] == "MISSING_CPU_ISA"
    assert "AVX512F" in result["missing_flags"]
