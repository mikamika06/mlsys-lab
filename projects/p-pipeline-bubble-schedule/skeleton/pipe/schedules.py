def generate_gpipe_schedule(num_stages: int, num_microbatches: int) -> list:
    raise NotImplementedError

def generate_1f1b_schedule(num_stages: int, num_microbatches: int) -> list:
    raise NotImplementedError

def generate_interleaved_1f1b_schedule(num_stages: int, num_microbatches: int, num_chunks: int) -> list:
    raise NotImplementedError

def generate_zero_bubble_schedule(num_stages: int, num_microbatches: int) -> list:
    raise NotImplementedError
