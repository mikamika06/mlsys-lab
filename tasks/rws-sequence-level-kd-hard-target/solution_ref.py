import numpy as np

def seq_level_kd_hard(teacher_logits, student_logits):
    """
    Compute sequence-level KD loss with hard targets.

    Hard target at each position = teacher's argmax token.
    Loss = mean cross-entropy of student against those targets.
    """
    T = np.asarray(teacher_logits, dtype=np.float64)
    S = np.asarray(student_logits, dtype=np.float64)

    targets = np.argmax(T, axis=-1)

    # numerically stable log-softmax via log-sum-exp
    shifted = S - np.max(S, axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))

    n = S.shape[0]
    loss = -np.mean(log_probs[np.arange(n), targets])
    return float(loss)
