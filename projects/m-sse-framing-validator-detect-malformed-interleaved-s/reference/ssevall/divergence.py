def locate_divergence_tokens(chat_prompt, completion_prompt):
    chat_tokens = chat_prompt.split()
    comp_tokens = completion_prompt.split()

    divergence_index = 0
    for i, (c, p) in enumerate(zip(chat_tokens, comp_tokens)):
        if c != p:
            divergence_index = i
            break
    else:
        divergence_index = min(len(chat_tokens), len(comp_tokens))

    return {
        "divergence_index": divergence_index,
        "chat_template_prefix": chat_tokens[:divergence_index],
        "completion_prefix": comp_tokens[:divergence_index]
    }
