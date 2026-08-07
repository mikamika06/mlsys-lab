def schedule_adapter_batches(requests):
    batches = {}
    for req in requests:
        aid = req.get("adapter_id", 0)
        batches.setdefault(aid, []).append(req)
    return list(batches.values())
