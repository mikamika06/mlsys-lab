import sys

sys.path.insert(0, ".")
from blobstore import build_blob_index, unique_bytes_on_disk, naive_total_bytes, incremental_pull_bytes

CONFIG = {"tags": {
    "modelA:fp16": [
        {"digest": "sha256:weightsA", "size": 1000},
        {"digest": "sha256:tok-shared", "size": 200},
    ],
    "modelA:q4": [
        {"digest": "sha256:weightsA-q4", "size": 200},
        {"digest": "sha256:tok-shared", "size": 200},
    ],
    "modelB:fp16": [
        {"digest": "sha256:weightsB", "size": 1000},
    ],
}}
BY_DIGEST = {}
for _tag, _blobs in CONFIG["tags"].items():
    for _b in _blobs:
        BY_DIGEST[_b["digest"]] = _b["size"]


def test_every_digest_appears_exactly_once():
    idx = build_blob_index(CONFIG)
    digests = [e["digest"] for e in idx]
    assert sorted(digests) == sorted(BY_DIGEST), f"{sorted(digests)} != {sorted(BY_DIGEST)}"
    assert len(digests) == len(set(digests)), "a digest appears in more than one entry"


def test_entry_size_matches_its_own_digest():
    idx = build_blob_index(CONFIG)
    for e in idx:
        assert e["size"] == BY_DIGEST[e["digest"]], \
            f"entry {e['digest']} reports size {e['size']}, expected {BY_DIGEST[e['digest']]}"


def test_dedup_never_costs_more_than_naive():
    a = unique_bytes_on_disk(CONFIG)
    b = naive_total_bytes(CONFIG)
    assert a <= b, f"unique bytes {a} exceed naive total {b}"


def test_incremental_pull_within_bounds():
    candidate = [
        {"digest": "sha256:weightsA", "size": 1000},
        {"digest": "sha256:new-only", "size": 555},
    ]
    got = incremental_pull_bytes(CONFIG, candidate)
    total = sum(b["size"] for b in candidate)
    assert 0 <= got <= total, f"incremental {got} out of bounds [0, {total}]"
    assert got == 555, f"only the new blob (555) should cost bytes, got {got}"
