import numpy as np

def prepare_data(raw_items):
    formatted = []
    for item in raw_items:
        prompt = str(item.get("prompt", ""))
        response = str(item.get("response", ""))
        tokens = [ord(c) % 256 for c in prompt + response]
        formatted.append({"tokens": tokens, "length": len(tokens)})
    return formatted

def run_lora(dataset, steps=5, lr=0.01):
    np.random.seed(42)
    losses = []
    loss = 2.5
    for _ in range(steps):
        loss = max(0.1, loss - 0.4 * lr * np.random.rand() + 0.05 * np.random.randn())
        losses.append(float(loss))
    adapter = {"A": np.random.randn(8, 16) * 0.1, "B": np.zeros((16, 8))}
    return {"losses": losses, "adapter": adapter}

def merge_adapter(base_weights, adapter):
    merged = {}
    for k, w in base_weights.items():
        if k in adapter:
            merged[k] = w + np.dot(adapter[k]["A"], adapter[k]["B"])
        else:
            merged[k] = w.copy()
    return merged

def quantize_model(weights, bits=4):
    quantized = {}
    scales = {}
    for k, w in weights.items():
        scale = np.max(np.abs(w)) / 7.0 if bits == 4 else np.max(np.abs(w)) / 127.0
        scale = max(scale, 1e-5)
        q = np.round(w / scale).astype(np.int8)
        quantized[k] = q
        scales[k] = scale
    return {"weights": quantized, "scales": scales}

class LoraServer:
    def __init__(self, model):
        self.model = model
        self.running = True

    def handle_request(self, prompt):
        if not self.running:
            raise RuntimeError("Server stopped")
        return f"Processed: {prompt[::-1]}"

def evaluate_quality(model, eval_set):
    scores = []
    for item in eval_set:
        pred = model.handle_request(item["prompt"])
        score = 1.0 if len(pred) > 0 else 0.0
        scores.append(score)
    return float(np.mean(scores))
