import sys
sys.path.insert(0, ".")
from router.admission import admit

def test_admit_accounts_for_generated_tokens():
    queue = [{"id": "new", "input_len": 50}]
    active = [{"id": "existing", "input_len": 100, "generated_len": 60}]

    result = admit(
        queue=queue,
        active=active,
        max_total_tokens=200,
        max_prefill_tokens=100,
        waiting_served_ratio=0.0
    )

    assert result == []
