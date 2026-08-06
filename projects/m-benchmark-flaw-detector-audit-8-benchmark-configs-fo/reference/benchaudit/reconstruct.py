def reconstruct_cli(result_data: dict) -> str:
    """Reconstruct vllm bench serve invocation from metadata."""
    cmd = ["vllm", "bench", "serve"]
    if "model" in result_data:
        cmd.extend(["--model", str(result_data["model"])])
    if "backend" in result_data:
        cmd.extend(["--backend", str(result_data["backend"])])
    if "endpoint" in result_data:
        cmd.extend(["--endpoint", str(result_data["endpoint"])])
    if "request_rate" in result_data:
        cmd.extend(["--request-rate", str(result_data["request_rate"])])
    if "num_prompts" in result_data:
        cmd.extend(["--num-prompts", str(result_data["num_prompts"])])
    if "dataset_name" in result_data:
        cmd.extend(["--dataset-name", str(result_data["dataset_name"])])
    if "max_concurrency" in result_data:
        cmd.extend(["--max-concurrency", str(result_data["max_concurrency"])])
    if result_data.get("ignore_eos", False):
        cmd.append("--ignore-eos")
    return " ".join(cmd)
