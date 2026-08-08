def locate_divergence_tokens(chat_prompt: str, comp_prompt: str) -> list[str]:
    chat_tokens = chat_prompt.split()
    comp_tokens = comp_prompt.split()
    divergent = []
    for t in chat_tokens:
        if t not in comp_tokens:
            divergent.append(t)
    return divergent
