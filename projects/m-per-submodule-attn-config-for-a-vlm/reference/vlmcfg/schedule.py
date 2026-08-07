def submodule_schedule(window, block_size, total_steps):
    schedule = []
    for step in range(total_steps):
        if window is None:
            freed = 0
        else:
            active = min(step + 1, window)
            blocks_needed = (active + block_size - 1) // block_size
            max_blocks = (window + block_size - 1) // block_size
            freed = max(0, blocks_needed - max_blocks // 2)
        schedule.append(freed)
    return schedule
