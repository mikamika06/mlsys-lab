class Request:
    def __init__(self, req_id, arrival, prompt_len, decode_len):
        self.id = req_id
        self.arrival = arrival
        self.prompt_len = prompt_len
        self.decode_len = decode_len


def simulate_static(requests, max_batch_size):
    raise NotImplementedError


def simulate_continuous(requests, max_batch_size):
    raise NotImplementedError


def compare_throughput(requests, max_batch_size):
    raise NotImplementedError
