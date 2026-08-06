import ref


def check(workdir):
    from loraeval.oom import tune_training_batch
    fixtures = ref.get_m2_fixtures()
    ok = 0
    for fx in fixtures:
        res = tune_training_batch(
            fx["vram_limit_mb"],
            fx["base_mb"],
            fx["rank"],
            fx["seq_len"],
            fx["target_tokens"]
        )
        if isinstance(res, dict) and "batch_size" in res and "iterations" in res:
            if res["batch_size"] > 0 and res["iterations"] > 0:
                ok += 1
    return {"oom_solved": 1.0 if ok == len(fixtures) else 0.0}
