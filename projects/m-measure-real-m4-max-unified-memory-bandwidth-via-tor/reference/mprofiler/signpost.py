def count_kernel_launches(log_lines):
    count = 0
    for line in log_lines:
        if "os_signpost" in line or "kernel_launch" in line or "Metal" in line:
            count += 1
    return count
