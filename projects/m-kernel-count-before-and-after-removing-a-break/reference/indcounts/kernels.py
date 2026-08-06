import re


def count_kernels(code_str):
    matches = re.findall(r"@triton\.jit|def triton_", code_str)
    return len(matches)


def count_before_and_after(code_str_before, code_str_after):
    return {
        "before": count_kernels(code_str_before),
        "after": count_kernels(code_str_after)
    }
