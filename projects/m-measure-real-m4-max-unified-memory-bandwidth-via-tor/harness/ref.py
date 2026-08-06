TEST_CASES = [
    {"bytes": 273 * 10**9, "duration": 0.5, "peak": 546.0},
    {"bytes": 100 * 10**9, "duration": 1.0, "peak": 546.0},
    {"bytes": 546 * 10**9, "duration": 1.0, "peak": 546.0},
]

TIMELINE_INPUTS = (
    [1024, 2048, 4096],
    [2048, 4096, 8192]
)

LOG_INPUTS = [
    ["os_signpost: start kernel_launch", "info: text", "os_signpost: end kernel_launch"],
    ["no match here"],
    ["Metal kernel_launch id=1", "Metal kernel_launch id=2"]
]
