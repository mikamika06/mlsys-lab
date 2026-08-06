def fold_rmsnorm_gamma(W, b, gamma):
    W_folded = []
    for row in W:
        new_row = [w * g for w, g in zip(row, gamma)]
        W_folded.append(new_row)
    b_folded = list(b)
    return W_folded, b_folded
