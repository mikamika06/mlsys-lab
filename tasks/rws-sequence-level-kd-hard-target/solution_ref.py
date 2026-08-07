import math

def seq_level_kd_hard(teacher_logits: list[list[float]], student_logits: list[list[float]]) -> float:
    """
    Compute sequence-level KD loss with hard targets.

    Hard target at each position = teacher's argmax token.
    Loss = mean cross-entropy of student against those targets.
    """
    T = teacher_logits
    S = student_logits

    n = len(S)
    V = len(S[0])

    targets = [0] * n
    for i in range(n):
        max_val = T[i][0]
        max_idx = 0
        for j in range(1, V):
            if T[i][j] > max_val:
                max_val = T[i][j]
                max_idx = j
        targets[i] = max_idx

    log_probs = [[0.0] * V for _ in range(n)]
    for i in range(n):
        max_s = S[i][0]
        for j in range(1, V):
            if S[i][j] > max_s:
                max_s = S[i][j]

        sum_exp = 0.0
        for j in range(V):
            sum_exp += math.exp(S[i][j] - max_s)
        log_sum_exp = max_s + math.log(sum_exp)

        for j in range(V):
            log_probs[i][j] = S[i][j] - log_sum_exp

    total_ce = 0.0
    for i in range(n):
        total_ce += -log_probs[i][targets[i]]

    loss = total_ce / float(n)
    return float(loss)
