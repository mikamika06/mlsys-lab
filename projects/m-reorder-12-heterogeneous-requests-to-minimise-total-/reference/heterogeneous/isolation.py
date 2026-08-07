class RunnerState:
    def __init__(self):
        self.num_ctx = 2048
        self.active_request = None

def execute_request(runner, request):
    original_ctx = runner.num_ctx
    if "num_ctx" in request:
        runner.num_ctx = request["num_ctx"]
    try:
        result = f"processed_with_ctx_{runner.num_ctx}"
    finally:
        runner.num_ctx = original_ctx
    return result
