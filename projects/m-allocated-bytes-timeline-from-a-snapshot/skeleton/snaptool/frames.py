def find_retaining_frame(snapshot):
    """
    Analyzes un-freed allocations in the snapshot trace to identify the top stack frame
    retaining the largest amount of memory.
    Returns (frame_string, total_bytes_retained) where frame_string is 'filename:line:name'.
    """
    raise NotImplementedError
