from sharder.plan import build_shard_plan
from sharder.vocab import export_vocab_only

def compute_conversion_schedule(vocab_data, tensors, max_bytes):
    vocab_meta = export_vocab_only(vocab_data)
    shards = build_shard_plan(tensors, max_bytes)
    return {"vocab": vocab_meta, "shards": shards}
