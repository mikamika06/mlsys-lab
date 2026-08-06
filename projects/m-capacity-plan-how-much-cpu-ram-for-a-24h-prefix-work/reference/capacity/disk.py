import math

def measure_read_amplification(block_size_bytes: int, fetch_requests: list[dict]) -> dict:
    logical_bytes = 0
    physical_bytes = 0
    
    for req in fetch_requests:
        offset = req["offset_bytes"]
        length = req["length_bytes"]
        logical_bytes += length
        
        start_block = offset // block_size_bytes
        end_block = (offset + length - 1) // block_size_bytes
        num_blocks = (end_block - start_block) + 1
        physical_bytes += num_blocks * block_size_bytes
        
    amp_factor = (physical_bytes / logical_bytes) if logical_bytes > 0 else 1.0
    return {
        "logical_bytes": logical_bytes,
        "physical_bytes": physical_bytes,
        "read_amplification": amp_factor
    }
