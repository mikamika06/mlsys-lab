def moe_all_to_all_dispatch(X: list[list[float]], router_logits: list[list[float]],
                             expert_weight: list[list[list[float]]], num_devices: int):
    """Simulate expert-parallel MoE dispatch: route -> all-to-all -> expert
    compute -> all-to-all back.

    See task.md for the exact routing/placement rule.
    """
    N = len(X)
    d = len(X[0])
    E = len(router_logits[0])
    experts_per_device = E // num_devices

    expert_id = [0] * N
    for i in range(N):
        best_e = 0
        best_val = router_logits[i][0]
        for e in range(1, E):
            val = router_logits[i][e]
            if val > best_val:
                best_val = val
                best_e = e
        expert_id[i] = best_e

    device_id = [0] * N
    for i in range(N):
        device_id[i] = expert_id[i] // experts_per_device

    device_counts = [0] * num_devices
    for i in range(N):
        device_counts[device_id[i]] += 1

    output = [[0.0] * d for _ in range(N)]

    for dev in range(num_devices):
        idx = []
        for i in range(N):
            if device_id[i] == dev:
                idx.append(i)

        if len(idx) == 0:
            continue

        dev_tokens_list = []
        dev_experts_list = []
        for i in idx:
            dev_tokens_list.append(X[i])
            dev_experts_list.append(expert_id[i])

        dev_tokens = dev_tokens_list
        dev_experts = dev_experts_list

        dev_out = [[0.0] * d for _ in range(len(idx))]

        unique_experts = []
        for e in dev_experts:
            found = False
            for ue in unique_experts:
                if ue == e:
                    found = True
                    break
            if not found:
                unique_experts.append(e)

        for e in unique_experts:
            m = []
            for j in range(len(dev_experts)):
                m.append(dev_experts[j] == e)

            w = expert_weight[e]
            for j in range(len(dev_tokens)):
                if m[j]:
                    row_res = [0.0] * d
                    tok = dev_tokens[j]
                    for r in range(d):
                        acc = 0.0
                        for c in range(d):
                            acc += tok[c] * w[c][r]
                        row_res[r] = acc
                    dev_out[j] = row_res

        for k, i in enumerate(idx):
            output[i] = dev_out[k]

    return output, device_counts
