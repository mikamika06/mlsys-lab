def select_runner(spec: dict) -> dict:
    """Selects the optimal local runner based on deployment constraints."""
    lf_score = 0
    ol_score = 0
    reasons = []

    if spec.get("single_file_dist"):
        lf_score += 30
        ol_score -= 30
        reasons.append("Single-file distribution favors llamafile.")

    if spec.get("air_gapped"):
        lf_score += 20
        ol_score += 10
        reasons.append("Air-gapped deployment supported by both runners.")

    os_targets = spec.get("os_targets", [])
    if len(os_targets) > 1:
        lf_score += 20
        reasons.append("Multi-OS target environment favors Cosmopolitan llamafile binary.")

    if spec.get("api_standard") == "cli_embedded":
        lf_score += 15
    elif spec.get("api_standard") == "openai_http":
        ol_score += 25

    if spec.get("multi_model_serving"):
        ol_score += 35
        lf_score -= 25
        reasons.append("Multi-model concurrent serving favors Ollama daemon.")

    rec = "llamafile" if lf_score >= ol_score else "ollama"

    return {
        "recommended_runner": rec,
        "llamafile_score": lf_score,
        "ollama_score": ol_score,
        "reasons": reasons,
    }
