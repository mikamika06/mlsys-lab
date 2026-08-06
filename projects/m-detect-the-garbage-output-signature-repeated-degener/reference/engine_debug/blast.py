def calculate_blast_radius(batch_requests: list[dict], crashed_request_ids: list[str]) -> dict:
    crashed_set = set(crashed_request_ids)
    total_tokens_lost = 0
    requests_lost = 0
    retry_list = []
    for req in batch_requests:
        if req["id"] in crashed_set:
            requests_lost += 1
            total_tokens_lost += req.get("generated_tokens_count", 0)
            retry_list.append({
                "id": req["id"],
                "prompt_tokens": req.get("prompt_tokens", []),
                "max_tokens": req.get("max_tokens", 128)
            })
    return {
        "requests_lost": requests_lost,
        "total_tokens_lost": total_tokens_lost,
        "retry_requests": retry_list
    }
