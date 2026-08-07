def predict_itl_jitter(prefill_lens, budget, decodes_per_step, base_itl):
    jitting = []
    for l in prefill_lens:
        rem = budget - decodes_per_step
        if rem <= 0:
            jitting.append(base_itl * 10.0)
            continue
        spike = base_itl * (1.0 + float(l) / float(budget))
        jitting.append(spike)
    return jitting
