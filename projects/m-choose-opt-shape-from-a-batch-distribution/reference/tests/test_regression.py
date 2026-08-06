from trtprof.runtime import ProfileRuntimeEngine


def test_out_of_profile_enqueue():
    profiles = [
        {
            "input_ids": {
                "min": (1, 8),
                "opt": (4, 32),
                "max": (8, 64)
            }
        }
    ]
    engine = ProfileRuntimeEngine(profiles)
    
    valid_res = engine.safe_enqueue({"input_ids": (4, 32)})
    assert valid_res["status"] == "ok"
    assert valid_res["profile_index"] == 0
    
    failed = False
    try:
        engine.safe_enqueue({"input_ids": (16, 128)})
    except ValueError:
        failed = True
    
    assert failed, "Engine failed to catch out-of-profile shape during enqueue"
