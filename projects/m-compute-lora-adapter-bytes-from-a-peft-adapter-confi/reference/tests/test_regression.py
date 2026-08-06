from loraserve.scheduler import schedule_adapter_batch


def test_adapter_scheduler_switches():
    requests = [
        {"id": 1, "adapter_id": "adapter_a"},
        {"id": 2, "adapter_id": "adapter_b"},
        {"id": 3, "adapter_id": "adapter_a"},
        {"id": 4, "adapter_id": "adapter_b"},
        {"id": 5, "adapter_id": "adapter_c"},
    ]
    batches = schedule_adapter_batch(requests, max_batch_size=2, max_active_adapters=1)

    assert len(batches) == 3
    assert [r["id"] for r in batches[0]] == [1, 3]
    assert [r["id"] for r in batches[1]] == [2, 4]
    assert [r["id"] for r in batches[2]] == [5]
