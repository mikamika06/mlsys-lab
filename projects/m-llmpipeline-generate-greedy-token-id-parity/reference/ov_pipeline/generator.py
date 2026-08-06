import numpy as np


def greedy_generate_handrolled(runner, prompt_ids, max_new_tokens):
    """Generates tokens greedily using a hand-rolled decoder loop."""
    current_ids = list(prompt_ids)
    for _ in range(max_new_tokens):
        logits = runner.step_logits(current_ids)
        next_token = int(np.argmax(logits))
        current_ids.append(next_token)
    return current_ids[len(prompt_ids):]


def greedy_generate_pipeline(runner, prompt_ids, max_new_tokens):
    """Generates tokens greedily using the pipeline interface."""
    return runner.pipeline_generate(prompt_ids, max_new_tokens)


def check_token_parity(runner, prompts, max_new_tokens):
    """Checks if greedy_generate_pipeline matches greedy_generate_handrolled."""
    for prompt in prompts:
        pipeline_tokens = greedy_generate_pipeline(runner, prompt, max_new_tokens)
        handrolled_tokens = greedy_generate_handrolled(runner, prompt, max_new_tokens)
        if pipeline_tokens != handrolled_tokens:
            return False
    return True
