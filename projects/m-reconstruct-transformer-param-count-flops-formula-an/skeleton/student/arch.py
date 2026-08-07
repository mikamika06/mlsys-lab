def compute_params(config):
    raise NotImplementedError


def compute_flops(config, seq_len):
    raise NotImplementedError


def compression_ratio(teacher_config, student_config):
    raise NotImplementedError
