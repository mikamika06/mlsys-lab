def generate_context_with_fact(length: int, position_ratio: float, fact: str) -> str:
    filler = "The quick brown fox jumps over the lazy dog. "
    words_needed = max(1, length // len(filler.split()))
    text_list = [filler] * words_needed
    insert_idx = int(len(text_list) * position_ratio)
    text_list.insert(insert_idx, fact)
    return "".join(text_list)
