class Request:
    def __init__(self, req_id, prompt_len, gen_len, priority=0):
        raise NotImplementedError

def simulate_scheduler(requests, policy="fcfs", max_num_seqs=4):
    raise NotImplementedError

def compare_policies(requests, target_req_id):
    raise NotImplementedError
