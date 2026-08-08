def find_np_saturation(
    total_ctx: int, req_slot_ctx: int, gpu_slot_cap: int = 64
) -> dict:
    """Find max parallel slots before context per slot drops below minimum or GPU slot cap."""
    if req_slot_ctx <= 0 or total_ctx <= 0:
        return {
            "sat_np": 0,
            "slot_ctx": 0,
            "wasted_ctx": total_ctx,
            "is_saturated": True,
            "max_np_by_ctx": 0,
        }
    max_np_by_ctx = total_ctx // req_slot_ctx
    sat_np = min(max_np_by_ctx, gpu_slot_cap)
    if sat_np < 1:
        return {
            "sat_np": 0,
            "slot_ctx": 0,
            "wasted_ctx": total_ctx,
            "is_saturated": True,
            "max_np_by_ctx": max_np_by_ctx,
        }
    slot_ctx = total_ctx // sat_np
    wasted_ctx = total_ctx - (slot_ctx * sat_np)
    is_saturated = (sat_np >= gpu_slot_cap) or (sat_np == max_np_by_ctx)
    return {
        "sat_np": sat_np,
        "slot_ctx": slot_ctx,
        "wasted_ctx": wasted_ctx,
        "is_saturated": is_saturated,
        "max_np_by_ctx": max_np_by_ctx,
    }
