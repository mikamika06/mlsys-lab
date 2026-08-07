def vocab_parallel_forward(token_ids: list[int], embedding: list[list[float]], output_weight: list[list[float]], world_size: int) -> tuple[list[list[float]], list[list[float]]]:
    vocab = len(embedding)
    hidden = len(embedding[0])
    n = len(token_ids)

    hidden_out = [[0.0 for _ in range(hidden)] for _ in range(n)]
    logits = [[0.0 for _ in range(vocab)] for _ in range(n)]

    bounds = []
    for i in range(world_size + 1):
        bounds.append(i * vocab // world_size)

    for rank in range(world_size):
        start = bounds[rank]
        end = bounds[rank + 1]

        local_hidden = [[0.0 for _ in range(hidden)] for _ in range(n)]
        for i in range(n):
            tid = token_ids[i]
            if start <= tid < end:
                for h in range(hidden):
                    local_hidden[i][h] = embedding[tid][h]

        for i in range(n):
            for h in range(hidden):
                hidden_out[i][h] += local_hidden[i][h]

        local_logits = [[0.0 for _ in range(vocab)] for _ in range(n)]
        for i in range(n):
            for v_idx in range(start, end):
                dot = 0.0
                for h in range(hidden):
                    dot += hidden_out[i][h] * output_weight[v_idx][h]
                local_logits[i][v_idx] = dot

        for i in range(n):
            for v in range(vocab):
                logits[i][v] += local_logits[i][v]

    return hidden_out, logits
