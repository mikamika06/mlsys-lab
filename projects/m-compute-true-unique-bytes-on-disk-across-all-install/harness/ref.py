CONFIGS = [
    {"tags": {
        "llama3:8b-fp16": [
            {"digest": "sha256:w-llama3-8b-fp16", "size": 16000000000},
            {"digest": "sha256:tok-llama3", "size": 9000000},
            {"digest": "sha256:lic-llama3", "size": 1500},
        ],
        "llama3:8b-q4_0": [
            {"digest": "sha256:w-llama3-8b-q4_0", "size": 4500000000},
            {"digest": "sha256:tok-llama3", "size": 9000000},
            {"digest": "sha256:lic-llama3", "size": 1500},
        ],
    }},
    {"tags": {
        "mistral:7b-fp16": [
            {"digest": "sha256:w-mistral-7b-fp16", "size": 14000000000},
            {"digest": "sha256:tok-mistral", "size": 8000000},
        ],
    }},
    {"tags": {
        "phi3:3.8b-q4": [
            {"digest": "sha256:w-phi3-q4", "size": 2200000000},
            {"digest": "sha256:tok-phi3", "size": 3000000},
        ],
        "phi3:3.8b-q8": [
            {"digest": "sha256:w-phi3-q8", "size": 4100000000},
            {"digest": "sha256:tok-phi3", "size": 3000000},
        ],
        "phi3:3.8b-fp16": [
            {"digest": "sha256:w-phi3-fp16", "size": 7600000000},
            {"digest": "sha256:tok-phi3", "size": 3000000},
        ],
    }},
]


def build_blob_index(config):
    by_digest = {}
    for tag, blobs in config["tags"].items():
        for b in blobs:
            entry = by_digest.setdefault(
                b["digest"], {"digest": b["digest"], "size": b["size"], "tags": set()}
            )
            entry["tags"].add(tag)
    return [
        {"digest": d, "size": by_digest[d]["size"], "tags": sorted(by_digest[d]["tags"])}
        for d in sorted(by_digest)
    ]


def unique_bytes_on_disk(config):
    return sum(b["size"] for b in build_blob_index(config))


def naive_total_bytes(config):
    return sum(b["size"] for blobs in config["tags"].values() for b in blobs)


def incremental_pull_bytes(config, candidate):
    have = {b["digest"] for blobs in config["tags"].values() for b in blobs}
    seen = set()
    total = 0
    for b in candidate:
        d = b["digest"]
        if d in have or d in seen:
            continue
        seen.add(d)
        total += b["size"]
    return total


def find_orphaned_blobs(config, disk_blobs):
    referenced = {b["digest"] for blobs in config["tags"].values() for b in blobs}
    return sorted(d for d in disk_blobs if d not in referenced)


def orphaned_bytes(config, disk_blobs):
    return sum(disk_blobs[d] for d in find_orphaned_blobs(config, disk_blobs))


CASES = [
    {
        "config": CONFIGS[0],
        "candidate": [
            {"digest": "sha256:w-llama3-8b-q8_0", "size": 8200000000},
            {"digest": "sha256:tok-llama3", "size": 9000000},
            {"digest": "sha256:lic-llama3", "size": 1500},
        ],
        "disk_blobs": {
            "sha256:w-llama3-8b-fp16": 16000000000,
            "sha256:w-llama3-8b-q4_0": 4500000000,
            "sha256:tok-llama3": 9000000,
            "sha256:lic-llama3": 1500,
        },
    },
    {
        "config": CONFIGS[1],
        "candidate": [
            {"digest": "sha256:w-mistral-7b-q4_0", "size": 4000000000},
            {"digest": "sha256:tok-mistral-new", "size": 8000000},
        ],
        "disk_blobs": {
            "sha256:w-mistral-7b-fp16": 14000000000,
            "sha256:tok-mistral": 8000000,
            "sha256:orphan-old-mistral-v1": 3500000000,
        },
    },
    {
        "config": CONFIGS[2],
        "candidate": [
            {"digest": "sha256:w-phi3-q4", "size": 2200000000},
            {"digest": "sha256:tok-phi3", "size": 3000000},
        ],
        "disk_blobs": {
            "sha256:w-phi3-q4": 2200000000,
            "sha256:w-phi3-q8": 4100000000,
            "sha256:w-phi3-fp16": 7600000000,
            "sha256:tok-phi3": 3000000,
        },
    },
    {
        "config": CONFIGS[1],
        "candidate": [
            {"digest": "sha256:w-new-dup", "size": 1000000000},
            {"digest": "sha256:w-new-dup", "size": 1000000000},
            {"digest": "sha256:tok-mistral", "size": 8000000},
        ],
        "disk_blobs": {
            "sha256:w-mistral-7b-fp16": 14000000000,
            "sha256:tok-mistral": 8000000,
        },
    },
]
