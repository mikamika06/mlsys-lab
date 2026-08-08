import numpy as np

def generate_tasks(context_len, num_positions, needle="The secret password is apple."):
    positions = np.linspace(0.0, 1.0, num_positions)
    tasks = []
    rng = np.random.RandomState(42)
    haystack_words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "lorem", "ipsum", "dolor", "sit", "amet"]
    for pos in positions:
        total_words = context_len // 4
        words = rng.choice(haystack_words, size=total_words).tolist()
        insert_idx = int(pos * len(words))
        words.insert(insert_idx, needle)
        prompt = " ".join(words)
        tasks.append({"position": float(pos), "prompt": prompt, "needle": needle})
    return tasks
