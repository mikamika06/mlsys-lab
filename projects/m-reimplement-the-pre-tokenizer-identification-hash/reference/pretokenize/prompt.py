def check_double_bos(prompt_text: str, bos_token: str) -> bool:
    if not bos_token:
        return False
    occurrences = prompt_text.count(bos_token)
    if occurrences > 1:
        return True
    return prompt_text.startswith(bos_token + bos_token)
