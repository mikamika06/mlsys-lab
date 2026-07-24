def classify_roofline_region(configs):
    labels = []

    for batch, seq, d, machine_balance in configs:
        flops = 2.0 * batch * seq * seq * d
        flops += 2.0 * batch * seq * d * d

        bytes_moved = 4.0 * (batch * seq * d + batch * d * d)

        arithmetic_intensity = flops / bytes_moved

        if arithmetic_intensity < machine_balance:
            labels.append("bandwidth-bound")
        else:
            labels.append("compute-bound")

    return labels
