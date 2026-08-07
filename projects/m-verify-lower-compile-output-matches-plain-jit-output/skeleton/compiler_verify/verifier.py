def verify_compilation(x, expected_out, compiled_out, max_abs_err=1e-5):
    raise NotImplementedError


def inspect_stablehlo(text):
    raise NotImplementedError


def verify_export_roundtrip(x, expected_out, exported_out, max_abs_err=1e-5):
    raise NotImplementedError
