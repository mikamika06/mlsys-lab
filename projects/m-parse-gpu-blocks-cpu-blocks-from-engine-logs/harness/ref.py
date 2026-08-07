LOGS = [
    ("INFO 05-12 12:34:56 model_executor.py:123] # GPU blocks: 4096, # CPU blocks: 2048", {"gpu_blocks": 4096, "cpu_blocks": 2048}),
    ("DEBUG: # GPU blocks : 1024 , # CPU blocks : 512", {"gpu_blocks": 1024, "cpu_blocks": 512}),
    ("Initializing KV cache: GPU blocks = 2048, CPU blocks = 1024", {"gpu_blocks": 2048, "cpu_blocks": 1024}),
    ("Some other log line without blocks", {"gpu_blocks": 0, "cpu_blocks": 0}),
    ("vllm serving: # GPU blocks: 8192, # CPU blocks: 4096", {"gpu_blocks": 8192, "cpu_blocks": 4096}),
]

TP_LOGS = [
    ("# GPU blocks: 1000", "# GPU blocks: 1950", True),
    ("# GPU blocks: 1000", "# GPU blocks: 1200", False),
]
