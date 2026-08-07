def extract_kernel_code(model, inputs):
    raise NotImplementedError


def inspect_fusion(kernel_code):
    raise NotImplementedError


def analyze_fusion_gap(kernel_code, size_mode):
    raise NotImplementedError


def apply_compilation_controls(config_flags):
    raise NotImplementedError


def optimize_both_sizes(model, sizes):
    raise NotImplementedError
