import math

COMPARISON_PAIRS = [
    (
        {"tokenizer_id": "llama2", "context_length": 4096, "stride": 512, "dataset_hash": "h1"},
        {"tokenizer_id": "llama2", "context_length": 4096, "stride": 512, "dataset_hash": "h1"}
    ),
    (
        {"tokenizer_id": "llama2", "context_length": 4096, "stride": 512, "dataset_hash": "h1"},
        {"tokenizer_id": "llama3", "context_length": 4096, "stride": 512, "dataset_hash": "h1"}
    ),
    (
        {"tokenizer_id": "llama2", "context_length": 2048, "stride": 512, "dataset_hash": "h1"},
        {"tokenizer_id": "llama2", "context_length": 4096, "stride": 512, "dataset_hash": "h1"}
    ),
    (
        {"tokenizer_id": "llama2", "context_length": 4096, "stride": 256, "dataset_hash": "h1"},
        {"tokenizer_id": "llama2", "context_length": 4096, "stride": 512, "dataset_hash": "h2"}
    ),
]

HELLASWAG_TASKS = [
    [
        {"context": "A man is sitting.", "endings": ["He sleeps.", "He stands up.", "He cooks.", "He flies."], "label": 1},
        {"context": "A dog barks.", "endings": ["It meows.", "It runs.", "It reads.", "It sings."], "label": 1},
        {"context": "She opens a book.", "endings": ["She reads it.", "She eats it.", "She drives it.", "She paints it."], "label": 0},
        {"context": "The sun rises.", "endings": ["Night falls.", "It gets bright.", "It rains ice.", "It freezes."], "label": 1},
    ],
    [
        {"context": "Item 1", "endings": ["A", "B"], "label": 0},
        {"context": "Item 2", "endings": ["A", "B"], "label": 0},
        {"context": "Item 3", "endings": ["A", "B"], "label": 1},
        {"context": "Item 4", "endings": ["A", "B"], "label": 1},
        {"context": "Item 5", "endings": ["A", "B"], "label": 0},
    ],
    [
        {"context": "Single item", "endings": ["X", "Y"], "label": 0}
    ]
]


def check_comparison_validity(run_a, run_b):
    reasons = []
    if run_a.get("tokenizer_id") != run_b.get("tokenizer_id"):
        reasons.append("tokenizer_mismatch")
    if run_a.get("context_length") != run_b.get("context_length"):
        reasons.append("context_length_mismatch")
    if run_a.get("stride") != run_b.get("stride"):
        reasons.append("stride_mismatch")
    if run_a.get("dataset_hash") != run_b.get("dataset_hash"):
        reasons.append("dataset_mismatch")
    return {"valid": len(reasons) == 0, "reasons": reasons}


def is_statistically_significant(score_a, stderr_a, score_b, stderr_b, z_threshold=1.96):
    diff = abs(score_a - score_b)
    combined_err = math.sqrt(stderr_a ** 2 + stderr_b ** 2)
    if combined_err == 0.0:
        return diff > 0.0
    z_score = diff / combined_err
    return z_score >= z_threshold


def dummy_model_fn(ctx, endings):
    if "man" in ctx or "dog" in ctx or "book" in ctx or "sun" in ctx:
        if "stands" in endings[1] or "runs" in endings[1] or "reads" in endings[0] or "bright" in endings[1]:
            return [0.1, 0.9, 0.0, 0.0] if "reads" not in endings[0] else [0.9, 0.1, 0.0, 0.0]
    return [0.8, 0.2]


def run_hellaswag_eval(items, model_fn):
    if not items:
        return {"acc": 0.0, "stderr": 0.0, "count": 0}
    correct_list = []
    for item in items:
        ctx = item["context"]
        endings = item["endings"]
        label = item["label"]
        scores = model_fn(ctx, endings)
        pred = max(range(len(scores)), key=lambda i: scores[i])
        correct_list.append(1.0 if pred == label else 0.0)
    n = len(correct_list)
    acc = sum(correct_list) / n
    if n <= 1:
        stderr = 0.0
    else:
        var = sum((x - acc) ** 2 for x in correct_list) / (n - 1)
        stderr = math.sqrt(var / n)
    return {"acc": float(acc), "stderr": float(stderr), "count": n}
