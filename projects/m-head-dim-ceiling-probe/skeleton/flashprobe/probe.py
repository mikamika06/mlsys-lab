def probe_head_dim_ceiling(hardware_spec):
    raise NotImplementedError

def compare_throughput(version, head_dim, seq_len):
    raise NotImplementedError

def check_fp8_availability(head_dim, sm_version):
    raise NotImplementedError
