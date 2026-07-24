def classify_norm_wiring(blocks):
    """
    Classify the normalisation wiring of transformer blocks.

    Parameters
    ----------
    blocks : list[dict]
        Each dict must contain boolean keys 'pre_norm' and 'post_norm'.

    Returns
    -------
    list[str]
        One of 'pre', 'post', or 'sandwich' per block.
    """
    labels = []
    for idx, block in enumerate(blocks):
        pre = bool(block.get("pre_norm", False))
        post = bool(block.get("post_norm", False))

        if not (pre or post):
            raise ValueError(f"Block {idx} has neither pre_norm nor post_norm set")

        if pre and not post:
            labels.append("pre")
        elif post and not pre:
            labels.append("post")
        else:  # both True
            labels.append("sandwich")
    return labels
