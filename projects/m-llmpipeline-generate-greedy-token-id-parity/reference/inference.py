import numpy as np

def run_pipeline(pipeline, prompt_ids, max_tokens):
    return pipeline.generate(prompt_ids, max_tokens)

def run_hand_rolled(model, prompt_ids, max_tokens):
    prompt_ids = list(prompt_ids)
    generated = []
    for _ in range(max_tokens):
        logits = model.infer(prompt_ids)
        next_token = int(np.argmax(logits[0, -1, :]))
        generated.append(next_token)
        prompt_ids.append(next_token)
    return generated

def format_chat_safe(pipeline, messages):
    try:
        return pipeline.apply_chat_template(messages)
    except RuntimeError:
        return "\n".join(m["content"] for m in messages)
