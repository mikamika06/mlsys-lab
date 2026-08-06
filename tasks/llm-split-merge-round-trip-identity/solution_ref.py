def split_heads(x: list[list[list[float]]], num_heads: int) -> list[list[list[list[float]]]]:
    """
    Split the last dimension of x into `num_heads` heads.
    Input shape: (B, T, D)
    Output shape: (B, T, num_heads, D // num_heads)
    """
    batch_size = len(x)
    seq_len = len(x[0])
    dim = len(x[0][0])
    head_dim = dim // num_heads

    result = []
    for b in range(batch_size):
        b_list = []
        for t in range(seq_len):
            row = x[b][t]
            heads_row = []
            for h in range(num_heads):
                head_chunk = []
                for d in range(head_dim):
                    head_chunk.append(row[h * head_dim + d])
                heads_row.append(head_chunk)
            b_list.append(heads_row)
        result.append(b_list)
    return result

def merge_heads(heads: list[list[list[list[float]]]]) -> list[list[list[float]]]:
    """
    Merge the heads back into a single last dimension.
    Input shape: (B, T, num_heads, D // num_heads)
    Output shape: (B, T, D)
    """
    batch_size = len(heads)
    seq_len = len(heads[0])
    num_heads = len(heads[0][0])
    head_dim = len(heads[0][0][0])

    result = []
    for b in range(batch_size):
        b_list = []
        for t in range(seq_len):
            row = []
            for h in range(num_heads):
                for d in range(head_dim):
                    row.append(heads[b][t][h][d])
            b_list.append(row)
        result.append(b_list)
    return result
