import re


def parse_draft_acceptance_line(log_text):
    pattern = r"draft\s+acceptance\s+rate\s*:\s*([\d\.]+)\%\s*\(\s*(\d+)\s*/\s*(\d+)\s*tokens\s*\)"
    match = re.search(pattern, log_text, re.IGNORECASE)
    if not match:
        return None
    pct, accepted, sampled = match.groups()
    return {
        "acceptance_rate_pct": float(pct),
        "accepted_tokens": int(accepted),
        "sampled_tokens": int(sampled),
        "ratio": int(accepted) / int(sampled) if int(sampled) > 0 else 0.0
    }


def recompute_draft_accept_ratio(timings_per_token):
    accepted = sum(int(item.get("draft_accepted", 0)) for item in timings_per_token)
    sampled = sum(int(item.get("draft_sampled", 0)) for item in timings_per_token)
    ratio = accepted / sampled if sampled > 0 else 0.0
    return {
        "accepted_tokens": accepted,
        "sampled_tokens": sampled,
        "draft_accept_ratio": ratio
    }
