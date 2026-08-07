import math


class Request:

    def __init__(
        self, req_id, prompt_len, max_gen_len, arrival_time, priority=0
    ):
        self.req_id = req_id
        self.prompt_len = prompt_len
        self.max_gen_len = max_gen_len
        self.arrival_time = arrival_time
        self.priority = priority
        self.output_len = 0
        self.scheduled_time = None
        self.completion_time = None

    @property
    def total_len(self):
        return self.prompt_len + self.output_len

    @property
    def is_finished(self):
        return self.output_len >= self.max_gen_len


class Scheduler:

    def __init__(
        self, num_blocks, block_size, max_num_batched_tokens, policy="fcfs"
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.max_num_batched_tokens = max_num_batched_tokens
        self.policy = policy
        self.free_blocks = num_blocks
        self.waiting = []
        self.running = []
        self.completed = []

    def add_request(self, req):
        self.waiting.append(req)

    def _needed_blocks(self, tokens):
        if tokens == 0:
            return 0
        return math.ceil(tokens / self.block_size)

    def _sort_queue(self, queue):
        if self.policy == "priority":
            queue.sort(key=lambda r: (-r.priority, r.arrival_time, r.req_id))
        else:
            queue.sort(key=lambda r: (r.arrival_time, r.req_id))

    def step(self, current_step):
        self._sort_queue(self.running)

        i = 0
        while i < len(self.running):
            req = self.running[i]
            if req.is_finished:
                req.completion_time = current_step
                self.free_blocks += self._needed_blocks(req.total_len)
                self.completed.append(self.running.pop(i))
            else:
                curr_b = self._needed_blocks(req.total_len)
                next_b = self._needed_blocks(req.total_len + 1)
                needed = next_b - curr_b
                if needed > self.free_blocks:
                    preempted = self.running.pop(i)
                    self.free_blocks += curr_b
                    self.waiting.append(preempted)
                else:
                    self.free_blocks -= needed
                    i += 1

        self._sort_queue(self.waiting)

        curr_batched_tokens = sum(
            1 if r.scheduled_time is not None else r.prompt_len
            for r in self.running
        )

        i = 0
        while i < len(self.waiting):
            req = self.waiting[i]
            is_preempted = req.scheduled_time is not None
            req_tokens = 1 if is_preempted else req.prompt_len
            if is_preempted:
                curr_b = self._needed_blocks(req.total_len)
                next_b = self._needed_blocks(req.total_len + 1)
                req_blocks = next_b - curr_b
            else:
                req_blocks = self._needed_blocks(req.prompt_len)

            if (
                curr_batched_tokens + req_tokens <= self.max_num_batched_tokens
                and self.free_blocks >= req_blocks
            ):
                req = self.waiting.pop(i)
                if req.scheduled_time is None:
                    req.scheduled_time = current_step
                self.free_blocks -= req_blocks
                curr_batched_tokens += req_tokens
                self.running.append(req)
            else:
                i += 1

        for req in self.running:
            req.output_len += 1

    def run_simulation(self, requests):
        unhandled = list(requests)
        current_step = 0
        while unhandled or self.waiting or self.running:
            i = 0
            while i < len(unhandled):
                if unhandled[i].arrival_time <= current_step:
                    self.add_request(unhandled.pop(i))
                else:
                    i += 1
            self.step(current_step)
            current_step += 1
        return self.completed
