def interleaved_1f1b_bubble_fraction(stages, microbatches, virtual_stages):
    idle_slots = stages - 1
    active_slots = microbatches * virtual_stages
    return float(idle_slots / (active_slots + idle_slots))
