import re


def parse_deepspeed_log(log_text):
    stage_match = re.search(r"ZeRO\s+stage\s*:\s*(\d+)", log_text, re.IGNORECASE)
    stage = int(stage_match.group(1)) if stage_match else None

    partition_match = re.search(
        r"optimizer_partition\s*:\s*(True|False)", log_text, re.IGNORECASE
    )
    partitioned = (
        partition_match.group(1).lower() == "true" if partition_match else False
    )

    reduction_match = re.search(
        r"estimated\s+memory\s+reduction\s*:\s*([\d\.]+)", log_text, re.IGNORECASE
    )
    reduction = float(reduction_match.group(1)) if reduction_match else None

    return {
        "zero_stage": stage,
        "optimizer_partitioned": partitioned,
        "estimated_reduction": reduction,
    }
