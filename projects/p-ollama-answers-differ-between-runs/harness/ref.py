import hashlib

def measure_divergence(client, prompt, runs=10):
    results = [client.generate(prompt) for _ in range(runs)]
    unique_count = len(set(results))
    return unique_count / runs

def resolve_options(modelfile_opts, api_opts, req_opts):
    res = dict(modelfile_opts)
    res.update(api_opts)
    res.update(req_opts)
    return res

def get_missing_parameters(payload, engine_spec):
    missing = [k for k in engine_spec if k not in payload]
    return missing

def make_deterministic_payload(prompt, seed=42, temperature=0.0, num_predict=128):
    return {
        "prompt": prompt,
        "options": {
            "seed": seed,
            "temperature": temperature,
            "num_predict": num_predict
        }
    }

def run_multiple_hashes(client, prompt, runs=10):
    outputs = [client.generate(prompt) for _ in range(runs)]
    hashes = [hashlib.sha256(o.encode()).hexdigest() for o in outputs]
    return len(set(hashes)) == 1
