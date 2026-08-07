def calibrate_max_and_error(batches):
    all_vals_list = []
    for b in batches:
        for row in b:
            for val in row:
                all_vals_list.append(val)

    amax = 0.0
    for val in all_vals_list:
        v = val if val >= 0.0 else -val
        if v > amax:
            amax = v
    amax = float(amax)

    scale = amax / 127.0

    recon_vals_list = []
    for b in batches:
        for row in b:
            for val in row:
                val_scaled = val / scale
                rounded = round(val_scaled)
                if rounded < -127:
                    q_val = -127.0
                elif rounded > 127:
                    q_val = 127.0
                else:
                    q_val = float(rounded)
                recon_val = q_val * scale
                recon_vals_list.append(recon_val)

    mse_sum = 0.0
    n_elements = len(all_vals_list)
    for i in range(n_elements):
        diff = recon_vals_list[i] - all_vals_list[i]
        mse_sum += diff * diff
    mse = float(mse_sum / n_elements)

    return amax, mse
