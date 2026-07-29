def victim(running):
    return max(running, key=lambda s: (s.prompt_len + s.decoded, s.rid))


def should_admit(state: dict) -> bool:
    return (state["running"] < state["max_seqs"]
            and state["free_blocks"] >= state["blocks_needed"])
