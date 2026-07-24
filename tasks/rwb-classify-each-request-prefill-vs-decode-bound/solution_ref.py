def classify_prefill_decode(requests):
    """
    Classify each request as 'prefill' or 'decode'.

    Parameters
    ----------
    requests : list[dict]
        Each dict must contain integer keys 'prompt_len' and 'gen_len'.

    Returns
    -------
    list[str]
        Labels for every request.
    """
    labels = []
    for r in requests:
        p = int(r["prompt_len"])
        g = int(r["gen_len"])
        prefill_cost = p * p
        decode_cost  = g * (p + g)
        if prefill_cost >= decode_cost:
            labels.append("prefill")
        else:
            labels.append("decode")
    return labels
