def max_working_set(addrs, window_size):
    """Return the maximum number of distinct byte addresses in any contiguous subsequence
    of length `window_size` from `addrs`.

    Parameters
    ----------
    addrs : list[int] | tuple[int]
        Sequence of byte-addresses accessed.
    window_size : int
        Length of each sliding window.

    Returns
    -------
    int
        Maximum working‑set size over all windows.
    """
    raise NotImplementedError('your code here')
