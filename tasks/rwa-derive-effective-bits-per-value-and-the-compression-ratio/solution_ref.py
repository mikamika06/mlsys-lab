def effective_bits_per_value(nbits, group_size, scale_bits=16, zero_bits=0):
    effective_bpv = float(nbits) + (
        float(scale_bits) + float(zero_bits)
    ) / float(group_size)
    compression_ratio = effective_bpv / 16.0
    return effective_bpv, compression_ratio
