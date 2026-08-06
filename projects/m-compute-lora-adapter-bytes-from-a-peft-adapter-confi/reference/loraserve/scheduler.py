def schedule_adapter_batch(
    requests: list[dict],
    max_batch_size: int,
    max_active_adapters: int
) -> list[list[dict]]:
    pending = list(requests)
    batches = []

    while pending:
        current_batch = []
        active_adapters = set()

        i = 0
        while i < len(pending):
            if len(current_batch) >= max_batch_size:
                break

            req = pending[i]
            adapter_id = req.get("adapter_id")

            if adapter_id is None:
                current_batch.append(pending.pop(i))
            elif adapter_id in active_adapters:
                current_batch.append(pending.pop(i))
            elif len(active_adapters) < max_active_adapters:
                active_adapters.add(adapter_id)
                current_batch.append(pending.pop(i))
            else:
                i += 1

        if not current_batch:
            current_batch.append(pending.pop(0))

        batches.append(current_batch)

    return batches
