from ring_attn.balancer import RingBalancer

def get_reference_balancer(world_size=4, seq_len=64):
    return RingBalancer(world_size, seq_len)
