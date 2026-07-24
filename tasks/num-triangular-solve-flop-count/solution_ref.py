def triangular_solve_flops(n: int) -> int:
    count = 0
    for row in range(n):
        count += row
    return count
