def compute_batch_counts(lengths, max_length):
    padding_batches = len(lengths)
    current = 0
    packing_batches = 1
    for l in sorted(lengths, reverse=True):
        if current + l <= max_length:
            current += l
        else:
            packing_batches += 1
            current = l
    return {"padding_batches": padding_batches, "packing_batches": packing_batches}
