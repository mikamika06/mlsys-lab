def moe_all_to_all(send: list[list[list[list[float]]]], world_size: int) -> list[list[list[float]]]:
    received = []
    for dst in range(world_size):
        blocks = []
        for src in range(world_size):
            blocks.extend(send[src][dst])
        received.append(blocks)
    return received
