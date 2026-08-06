LOGS = [
    ("[rank 0] Watchdog caught collective execution timeout during operation broadcast", {"failed_rank": 0, "timeout_op": "broadcast", "has_timeout": True}),
    ("[rank 2] Watchdog caught collective execution timeout during operation allreduce", {"failed_rank": 2, "timeout_op": "allreduce", "has_timeout": True}),
    ("Normal log output without any failure", {"failed_rank": -1, "timeout_op": "", "has_timeout": False})
]

MEMBERSHIP_TESTS = [
    (4, [1], 3, {"active_ranks": [0, 2, 3], "mapping": {0: 0, 2: 1, 3: 2}, "world_size": 3}),
    (8, [3, 5], 6, {"active_ranks": [0, 1, 2, 4, 6, 7], "mapping": {0: 0, 1: 1, 2: 2, 4: 3, 6: 4, 7: 5}, "world_size": 6})
]
