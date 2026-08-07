import sys

sys.path.insert(0, ".")
from shards.manifest import merge_manifests, validate_filename
from shards.validate import check_shard_set


def test_validate_filename_correct():
    assert validate_filename("model-0001-of-0003.gguf") is True
    assert validate_filename("model-0003-of-0003.gguf") is True


def test_validate_filename_incorrect():
    assert validate_filename("model-000-of-0003.gguf") is False
    assert validate_filename("model-0001-of-0000.gguf") is False
    assert validate_filename("not-a-shard.gguf") is False


def test_merge_manifests_basic():
    m1 = {"version": 1, "size": 100, "tensors": {"t1": {"offset": 0}}}
    m2 = {"version": 1, "size": 200, "tensors": {"t2": {"offset": 0}}}
    merged = merge_manifests([m1, m2])
    assert merged["total_size"] == 300
    assert "t1" in merged["tensors"]
    assert "t2" in merged["tensors"]


def test_check_shard_set_valid():
    files = ["model-0001-of-0002.gguf", "model-0002-of-0002.gguf"]
    manifests = [{"size": 50}, {"size": 50}]
    assert check_shard_set(files, manifests) is True
