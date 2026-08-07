def fork_tokens_saved(base_tokens, num_branches, shared_prefix_len):
    saved_per_branch = shared_prefix_len
    total_saved = (num_branches - 1) * saved_per_branch
    return total_saved
