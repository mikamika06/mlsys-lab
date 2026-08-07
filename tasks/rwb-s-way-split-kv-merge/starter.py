import math

def merge_split_kv(partials):
    """Merge S independent split-KV attention partials into the final output.

    partials: list of S (m_s, l_s, acc_s) tuples, all for the same n
        query rows:
          m_s: (n,) local running max score.
          l_s: (n,) local softmax denominator.
          acc_s: (n, d) local sum_j exp(score_ij - m_s_i) * V_j.

    Returns the (n, d) merged attention output:
      m* = max_s m_s
      l* = sum_s l_s * exp(m_s - m*)
      acc* = sum_s acc_s * exp(m_s - m*)
      out = acc* / l*
    """
    raise NotImplementedError('your code here')
