def generate_fixtures():
    num_turns = 10
    sys_len = 500
    dyn_len = 10
    hist_len = 50

    sys_blocks = [f"s_{i}" for i in range(sys_len)]

    hist_turns = []
    dyn_turns = []
    for t in range(num_turns):
        hist_turns.append([f"h_{t}_{i}" for i in range(hist_len)])
        if t < 4:
            dyn_turns.append([f"d_constant_{i}" for i in range(dyn_len)])
        else:
            dyn_turns.append([f"d_dynamic_{t}_{i}" for i in range(dyn_len)])

    bad_log = []
    good_log = []

    hist_so_far = []
    hist_so_far_states = []
    for t in range(num_turns):
        hist_so_far.extend(hist_turns[t])
        hist_so_far_states.append(list(hist_so_far))

        bad_log.append(sys_blocks + dyn_turns[t] + hist_so_far)
        good_log.append(sys_blocks + hist_so_far + dyn_turns[t])

    return sys_blocks, hist_so_far_states, dyn_turns, bad_log, good_log

SYS_BLOCKS, HIST_STATES, DYN_TURNS, BAD_LOG, GOOD_LOG = generate_fixtures()

def simulate_processing(prompts):
    counts = []
    for i, p in enumerate(prompts):
        if i == 0:
            counts.append(len(p))
        else:
            prev = prompts[i-1]
            match_len = 0
            for a, b in zip(p, prev):
                if a == b:
                    match_len += 1
                else:
                    break
            counts.append(len(p) - match_len)
    return counts

def find_breaking_turn(counts):
    if len(counts) <= 1:
        return -1
    return max(range(1, len(counts)), key=lambda i: counts[i])

def build_turn_prompt(system, history, dynamic):
    return system + history + dynamic

def total_processed_blocks(prompts):
    return sum(simulate_processing(prompts))
