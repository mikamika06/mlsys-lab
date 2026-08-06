from tritondrain.timeout import derive_minimum_drain_timeout


def test_drain_timeout_accounts_for_queue_delay():
    config = {
        "queue_delay_ms": 25,
        "requests": [
            {"id": "req-1", "stage": "executing", "remaining_ms": 30},
            {"id": "req-2", "stage": "queued", "remaining_ms": 20},
        ],
    }
    timeout = derive_minimum_drain_timeout(config)
    assert timeout == 45
