class Runner:
    def __init__(self, base_config):
        self.config = dict(base_config)
        self.active_num_ctx = base_config.get("num_ctx", 2048)

    def run(self, request):
        local_ctx = request.get("num_ctx", self.active_num_ctx)
        res = {"executed_ctx": local_ctx}
        return res

def execute_request(runner, request):
    saved_ctx = runner.active_num_ctx
    try:
        if "num_ctx" in request:
            runner.active_num_ctx = request["num_ctx"]
        return runner.run(request)
    finally:
        runner.active_num_ctx = saved_ctx
