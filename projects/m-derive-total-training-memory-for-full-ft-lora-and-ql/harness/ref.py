PARAM_COUNTS = [1_000_000_000, 7_000_000_000, 13_000_000_000, 70_000_000_000]
METHODS = ["full", "lora", "qlora"]
BUDGETS = {
    "consumer_24gb": 24 * 1024 * 1024 * 1024,
    "enterprise_80gb": 80 * 1024 * 1024 * 1024,
    "cluster_320gb": 320 * 1024 * 1024 * 1024
}
