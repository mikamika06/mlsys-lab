def _oracle(slots):
    packed = {
        "token": [],
        "pos": [],
        "n_seq_id": [],
        "seq_id": [],
        "logits": [],
    }

    for slot in slots:
        packed["token"].append(int(slot["token"]))
        packed["pos"].append(int(slot["position"]))
        packed["n_seq_id"].append(1)
        packed["seq_id"].append([int(slot["seq_id"])])
        packed["logits"].append(bool(slot["wants_logits"]))

    return packed


def grade(sol, fx) -> dict:
    cases = [
        [
            {
                "token": 11,
                "position": 0,
                "seq_id": 2,
                "wants_logits": True,
            },
            {
                "token": 12,
                "position": 1,
                "seq_id": 4,
                "wants_logits": False,
            },
        ],
        [
            {
                "token": 99,
                "position": 17,
                "seq_id": 0,
                "wants_logits": False,
            },
        ],
        [
            {
                "token": 3,
                "position": 4,
                "seq_id": 7,
                "wants_logits": True,
            },
            {
                "token": 8,
                "position": 5,
                "seq_id": 7,
                "wants_logits": True,
            },
            {
                "token": 13,
                "position": 6,
                "seq_id": 9,
                "wants_logits": False,
            },
            {
                "token": 21,
                "position": 7,
                "seq_id": 1,
                "wants_logits": True,
            },
        ],
    ]

    ok = 1.0
    for slots in cases:
        expected = _oracle(slots)
        try:
            got = sol.pack_llama_batch(slots)
        except Exception:
            ok = 0.0
            break

        if got != expected:
            ok = 0.0
            break

    return {
        "exact_match": ok
    }
