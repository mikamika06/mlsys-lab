TRACES = [
    (
        [{"launch_delay": 200, "driver_wait": 100}, {"launch_delay": 150, "driver_wait": 50}],
        [{"launch_delay": 40, "driver_wait": 20}, {"launch_delay": 30, "driver_wait": 10}]
    ),
    (
        [{"launch_delay": 500, "driver_wait": 200}],
        [{"launch_delay": 100, "driver_wait": 40}]
    ),
    (
        [{"launch_delay": 80, "driver_wait": 20}, {"launch_delay": 100, "driver_wait": 50}],
        [{"launch_delay": 20, "driver_wait": 5}, {"launch_delay": 20, "driver_wait": 10}]
    )
]

SPACES = [
    (
        [(16, 16, 16), (32, 32, 16)],
        [2, 4],
        [1, 2, 4]
    ),
    (
        [(64, 64, 32), (16, 16, 16), (8, 8, 8)],
        [4, 8],
        [2, 4]
    ),
    (
        [(32, 16, 16), (64, 32, 16), (16, 32, 16)],
        [2],
        [1, 3]
    )
]
