## Context

Knowledge distillation transfers knowledge from a large teacher model to a smaller
student model. In *sequence-level* KD the teacher provides a full output sequence
rather than per-token soft distributions. The simplest variant uses **hard targets**:
at each position the teacher's greedy (argmax) token becomes a one-hot label and
the student is trained with standard cross-entropy.

Formally, let the teacher logits be $Z^T \in \mathbb{R}^{n \times V}$ and the student
logits be $Z^S \in \mathbb{R}^{n \times V}$, where $n$ is the sequence length and
$V$ the vocabulary size. The hard target at position $i$ is

$$y_i = \arg\max_{j} \; Z^T_{i,j}.$$

The sequence-level KD loss with hard targets is

$$\mathcal{L}_{\text{KD-hard}} \;=\; -\,\frac{1}{n}\sum_{i=1}^{n}
  \log \frac{\exp\!\bigl(Z^S_{i,\,y_i}\bigr)}
            {\sum_{j=1}^{V}\exp\!\bigl(Z^S_{i,j}\bigr)}
  \;=\; -\,\frac{1}{n}\sum_{i=1}^{n}
  \bigl[Z^S_{i,\,y_i} - \log\!\textstyle\sum_{j}\exp(Z^S_{i,j})\bigr].$$

A numerically stable implementation computes the log-softmax via the log-sum-exp
trick:

$$\log\text{softmax}(Z^S)_{i,j} \;=\; Z^S_{i,j}
  - \Bigl(m_i + \log\!\textstyle\sum_{j}\exp\!\bigl(Z^S_{i,j}-m_i\bigr)\Bigr),
  \qquad m_i = \max_j Z^S_{i,j}.$$

## Task

Implement `seq_level_kd_hard(teacher_logits, student_logits)`:

```python
import numpy as np

def seq_level_kd_hard(teacher_logits: np.ndarray,
                      student_logits: np.ndarray) -> float:
    """
    Return the sequence-level KD hard-target cross-entropy loss.

    Parameters
    ----------
    teacher_logits : np.ndarray, shape (n, V), dtype float64
    student_logits : np.ndarray, shape (n, V), dtype float64

    Returns
    -------
    float  –  the scalar loss value.
    """
```

Use vectorized NumPy operations only — no Python `for` loops over positions.
The result must be a plain Python `float`.

## Example

```python
import numpy as np
teacher = np.array([[ 2.0,  1.0,  0.1],
                     [-1.0,  3.0,  0.5]])
student = np.array([[ 1.0,  2.0,  0.0],
                     [ 0.0,  1.0,  3.0]])

loss = seq_level_kd_hard(teacher, student)
# teacher argmax tokens: [1, 1]
# student log-softmax:
#   row 0: [-0.3133,  0.6867, -0.3133]
#   row 1: [-2.9518, -1.9518,  0.0482]
# loss = -mean(0.6867, -1.9518) = 0.6326 (approx)
```

## What the gate checks

The gate computes a reference answer using an independent NumPy implementation
(manual log-sum-exp, not reusing `scipy.special.logsumexp`) and reports the
relative $L_2$ error:

$$\texttt{rel\_err} = \frac{|\hat{L} - L^*|}{|L^*|}.$$

The solution passes when $\texttt{rel\_err} < 10^{-6}$. Common mistakes
(averaging over the full matrix, forgetting the negation, using KL on soft
targets) produce errors orders of magnitude larger.
