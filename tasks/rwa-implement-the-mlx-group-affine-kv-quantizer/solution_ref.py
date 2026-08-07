def quantize_kv_group_affine(kv: list[list[list[float]]], kv_bits: int, kv_group_size: int) -> tuple[list, list, list]:
    if isinstance(kv, list) and len(kv) > 0 and isinstance(kv[0], list):
        q_list = []
        scales_list = []
        zeros_list = []
        for subkv in kv:
            q_sub, s_sub, z_sub = quantize_kv_group_affine(subkv, kv_bits, kv_group_size)
            q_list.append(q_sub)
            scales_list.append(s_sub)
            zeros_list.append(z_sub)
        return q_list, scales_list, zeros_list
    else:
        qmax = (1 << kv_bits) - 1
        groups = len(kv) // kv_group_size

        q_groups = []
        scales = []
        zeros = []

        for g_idx in range(groups):
            group_slice = kv[g_idx * kv_group_size : (g_idx + 1) * kv_group_size]
            mn = group_slice[0]
            mx = group_slice[0]
            for val in group_slice[1:]:
                if val < mn:
                    mn = val
                if val > mx:
                    mx = val

            sc = (mx - mn) / qmax
            if sc == 0.0:
                sc = 1.0

            z = round(-mn / sc)
            if z < 0:
                z = 0
            elif z > qmax:
                z = qmax

            q_group = []
            for val in group_slice:
                val_float = float(val)
                q_val = round(val_float / sc + z)
                if q_val < 0:
                    q_val = 0
                elif q_val > qmax:
                    q_val = qmax
                q_group.append(int(q_val))

            q_groups.append(q_group)
            scales.append([sc])
            zeros.append([z])

        return q_groups, scales, zeros
