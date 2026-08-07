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
        raise NotImplementedError

    def step(self, current_step):
        raise NotImplementedError

    def run_simulation(self, requests):
        raise NotImplementedError
