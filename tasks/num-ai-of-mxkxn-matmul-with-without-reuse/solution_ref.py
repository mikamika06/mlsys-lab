def ai_matmul(M, K, N):
    F = 2.0 * M * N * K
    M_no_reuse = 8.0 * (2 * M * N * K + M * N)
    M_full_reuse = 8.0 * (M * K + K * N + M * N)
    return F / M_no_reuse, F / M_full_reuse
