#!/usr/bin/env python3
"""Record the runtime artifacts the library-bound Part-2 units are built on.

A unit like "render a Go-template chat prompt exactly as the runner does" is only
worth doing against the template a real runner actually ships — the interesting
cases are the ones nobody would invent (`{{- if }}` whitespace trimming, the
tool-call branch, the harmony channels in gpt-oss). Same for safetensors: the
header is JSON, but the alignment rules and the offset arithmetic only bite on a
file some other tool wrote.

So nothing here is synthesised. Every artifact is lifted from software installed
on this machine — ollama's own blob store, the safetensors writer, MLX's own
module naming — and written into projects/_fixtures/ with a provenance record.
The learner then works against real bytes with no library installed at all.

    python3 tools/record_artifacts.py           # everything available
    python3 tools/record_artifacts.py --only ollama
"""
import argparse
import glob
import hashlib
import json
import os
import platform
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(ROOT, "projects", "_fixtures")
OLLAMA = os.path.expanduser("~/.ollama/models")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def record_ollama(out):
    """Chat templates, parameter blocks and system prompts, as ollama stores them."""
    mans = sorted(glob.glob(os.path.join(
        OLLAMA, "manifests", "registry.ollama.ai", "library", "*", "*")))
    if not mans:
        return {"skipped": "no ollama blob store"}
    os.makedirs(out, exist_ok=True)
    got, index = set(), []
    for p in mans:
        model = p.split("library/")[1]
        d = json.load(open(p))
        entry = {"model": model, "layers": {}}
        for layer in d["layers"]:
            kind = layer["mediaType"].split(".")[-1]
            if kind not in ("template", "params", "system"):
                continue
            blob = os.path.join(OLLAMA, "blobs", layer["digest"].replace(":", "-"))
            if not os.path.isfile(blob):
                continue
            name = f"{model.replace('/', '_')}.{kind}"
            if name not in got:
                with open(blob, "rb") as f:
                    body = f.read()
                with open(os.path.join(out, name), "wb") as f:
                    f.write(body)
                got.add(name)
            entry["layers"][kind] = {"file": name, "bytes": layer["size"],
                                     "digest": layer["digest"][7:23]}
        if entry["layers"]:
            index.append(entry)
    with open(os.path.join(out, "index.json"), "w") as f:
        json.dump({"source": "ollama local blob store", "models": index}, f, indent=2)
    return {"models": len(index), "files": len(got)}


def record_safetensors(out, gguf_truth):
    """A real .safetensors written by the reference writer, holding real weights."""
    try:
        import numpy as np
        from safetensors.numpy import save_file
    except ImportError:
        return {"skipped": "safetensors not installed"}
    if not os.path.isfile(gguf_truth):
        return {"skipped": "run record_gguf.py first"}
    import numpy as np
    z = np.load(gguf_truth)
    os.makedirs(out, exist_ok=True)
    # Mixed dtypes on purpose: the header is where a hand-written parser gets the
    # element size wrong, and bf16 has no numpy scalar type to fall back on.
    tensors = {}
    for i, k in enumerate(z.files):
        arr = z[k]
        tensors[k.replace(".", "_")] = arr if i % 2 == 0 else arr.astype(np.float16)
    path = os.path.join(out, "weights.safetensors")
    save_file(tensors, path, metadata={"format": "pt", "produced_by": "safetensors writer"})
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        head = json.loads(f.read(n))
    with open(os.path.join(out, "header_truth.json"), "w") as f:
        json.dump({"header_bytes": n, "header": head,
                   "file_bytes": os.path.getsize(path)}, f, indent=2)
    return {"tensors": len(tensors), "header_bytes": n,
            "file_bytes": os.path.getsize(path)}


def record_mlx(out):
    """MLX's own parameter naming for a transformer block, beside the HF names."""
    try:
        import mlx.core as mx
        import mlx.nn as nn
    except ImportError:
        return {"skipped": "mlx not installed"}

    class Block(nn.Module):
        def __init__(self, d, h, ff):
            super().__init__()
            self.attention = nn.MultiHeadAttention(d, h)
            self.ln1 = nn.RMSNorm(d)
            self.ln2 = nn.RMSNorm(d)
            self.linear1 = nn.Linear(d, ff, bias=False)
            self.linear2 = nn.Linear(d, ff, bias=False)
            self.linear3 = nn.Linear(ff, d, bias=False)

    b = Block(256, 8, 512)
    flat = {}

    def walk(prefix, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(f"{prefix}.{k}" if prefix else k, v)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(f"{prefix}.{i}", v)
        else:
            flat[prefix] = [int(x) for x in obj.shape]

    walk("", b.parameters())
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "mlx_param_tree.json"), "w") as f:
        json.dump({"module": "mlx.nn transformer block",
                   "mlx_version": mx.__version__,
                   "params": dict(sorted(flat.items()))}, f, indent=2)
    return {"params": len(flat), "mlx": mx.__version__}


def record_env(out):
    os.makedirs(out, exist_ok=True)
    env = {"platform": platform.platform(), "machine": platform.machine(),
           "python": sys.version.split()[0]}
    for cmd in (["sysctl", "-n", "machdep.cpu.brand_string"],
                ["sysctl", "-n", "hw.memsize"],
                ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
                ["sysctl", "-n", "hw.perflevel1.physicalcpu"]):
        try:
            env[cmd[-1]] = subprocess.run(
                cmd, capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:
            pass
    try:
        env["llama_bench_version"] = subprocess.run(
            ["llama-bench", "--version"], capture_output=True, text=True,
            timeout=20).stderr.strip().splitlines()[0]
    except Exception:
        pass
    with open(os.path.join(out, "host.json"), "w") as f:
        json.dump(env, f, indent=2)
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    jobs = {
        "ollama": lambda: record_ollama(os.path.join(FIX, "ollama")),
        "safetensors": lambda: record_safetensors(
            os.path.join(FIX, "safetensors"),
            os.path.join(FIX, "gguf", "dequantized_truth.npz")),
        "mlx": lambda: record_mlx(os.path.join(FIX, "mlx")),
        "env": lambda: record_env(FIX),
    }
    report = {}
    for name, fn in jobs.items():
        if a.only and a.only != name:
            continue
        try:
            report[name] = fn()
        except Exception as e:
            report[name] = {"error": f"{type(e).__name__}: {e}"[:160]}
        print(f"{name:14} {json.dumps(report[name])[:150]}")
    with open(os.path.join(FIX, "PROVENANCE.json"), "w") as f:
        json.dump(report, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
