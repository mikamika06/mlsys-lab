class UnsupportedArchitectureError(Exception):
    pass


def verify_logits_parity(hf_logits, ov_logits, rel_err_threshold=1e-3):
    raise NotImplementedError


def compare_export_metrics(model_name):
    raise NotImplementedError


def validate_architecture(arch):
    raise NotImplementedError
