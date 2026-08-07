from cacheplan.analyze import simulate_processing

def build_turn_prompt(system: list[str], history: list[str], dynamic: list[str]) -> list[str]:
    return system + history + dynamic

def total_processed_blocks(prompts: list[list[str]]) -> int:
    return sum(simulate_processing(prompts))
