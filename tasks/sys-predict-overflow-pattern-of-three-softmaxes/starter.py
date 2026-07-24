def classify_softmax_overflow(z) -> tuple:
    """
    Given a 1-D array of raw scores z, run three softmax implementations
    on it and report whether each one's output probability vector
    contains any inf or nan (i.e. is numerically unstable on this input):

      1. naive: exp(z) / sum(exp(z)), no max-subtraction.
      2. lse: exp(z - max(z)) / sum(exp(z - max(z))) -- the standard
         log-sum-exp-stabilized softmax.
      3. online: single-pass streaming softmax (Milakov & Gimelshein /
         FlashAttention style) that scans z once, maintaining a running
         max m and running sum s, rescaling s by exp(old_m - new_m)
         every time the running max updates, then in a second pass
         computes exp(z - final_m) / final_s.

    Return (naive_overflow, lse_overflow, online_overflow) as bools.
    """
    raise NotImplementedError('your code here')
