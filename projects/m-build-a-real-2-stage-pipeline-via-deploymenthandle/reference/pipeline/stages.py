import time


class StageOne:
    def __init__(self, handle=None):
        self.handle = handle

    def process(self, data):
        meta = {"stage_one_start": time.perf_counter_ns()}
        processed = [x * 2 for x in data]
        meta["stage_one_end"] = time.perf_counter_ns()
        return {"data": processed, "meta": meta}


class StageTwo:
    def process(self, input_payload):
        meta = input_payload.get("meta", {})
        meta["stage_two_start"] = time.perf_counter_ns()
        data = input_payload["data"]
        result = [x + 1 for x in data]
        meta["stage_two_end"] = time.perf_counter_ns()
        return {"result": result, "meta": meta}


class PipelineOrchestrator:
    def __init__(self, stage_one, stage_two):
        self.stage_one = stage_one
        self.stage_two = stage_two

    def run(self, payload):
        first_out = self.stage_one.process(payload)
        if self.stage_one.handle is not None:
            final_out = self.stage_one.handle.remote(first_out)
        else:
            final_out = self.stage_two.process(first_out)
        return final_out
