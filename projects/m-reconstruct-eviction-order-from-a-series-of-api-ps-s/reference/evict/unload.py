def force_unload_and_verify(model_id, api_ps_before, api_ps_after):
    before_has = any(m.get("id") == model_id for m in api_ps_before)
    after_has = any(m.get("id") == model_id for m in api_ps_after)
    return before_has and not after_has
