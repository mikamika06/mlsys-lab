import numpy as np


def _get_shingles(tokens, k=3):
    if len(tokens) < k:
        return {tuple(tokens)}
    return {tuple(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def ref_deduplicate_samples(samples, num_perm=128, threshold=0.8):
    if not samples:
        return []

    shingle_sets = [_get_shingles(s, k=3) for s in samples]

    prime = 4294967311
    rng = np.random.RandomState(42)
    a = rng.randint(1, prime, size=(num_perm,), dtype=np.uint64)
    b = rng.randint(0, prime, size=(num_perm,), dtype=np.uint64)

    signatures = []
    for s_set in shingle_sets:
        hashes = [hash(sh) & 0xFFFFFFFF for sh in s_set]
        hashes_arr = np.array(hashes, dtype=np.uint64).reshape(-1, 1)
        phashes = (a * hashes_arr + b) % prime
        sig = phashes.min(axis=0)
        signatures.append(sig)

    signatures = np.array(signatures)

    num_bands = 16
    r = num_perm // num_bands

    duplicates = set()
    n = len(samples)

    for band_idx in range(num_bands):
        start = band_idx * r
        end = start + r
        buckets = {}
        for i in range(n):
            if i in duplicates:
                continue
            band_sig = tuple(signatures[i, start:end])
            buckets.setdefault(band_sig, []).append(i)

        for band_sig, candidate_indices in buckets.items():
            if len(candidate_indices) < 2:
                continue
            for idx1 in range(len(candidate_indices)):
                i = candidate_indices[idx1]
                if i in duplicates:
                    continue
                for idx2 in range(idx1 + 1, len(candidate_indices)):
                    j = candidate_indices[idx2]
                    if j in duplicates:
                        continue
                    s1, s2 = shingle_sets[i], shingle_sets[j]
                    intersection = len(s1.intersection(s2))
                    union = len(s1.union(s2))
                    sim = intersection / union if union > 0 else 1.0
                    if sim >= threshold:
                        duplicates.add(j)

    return [s for i, s in enumerate(samples) if i not in duplicates]


def ref_balance_lengths(samples, bucket_sizes, target_counts):
    sorted_buckets = sorted(bucket_sizes)
    buckets = {b: [] for b in sorted_buckets}

    for s in samples:
        length = len(s)
        for b in sorted_buckets:
            if length <= b:
                buckets[b].append(s)
                break

    result = []
    for b in sorted_buckets:
        target = target_counts.get(b, 0)
        available = buckets[b]
        if not available:
            continue
        if len(available) >= target:
            result.extend(available[:target])
        else:
            repeated = []
            while len(repeated) < target:
                repeated.extend(available)
            result.extend(repeated[:target])

    return result


def generate_dataset():
    rng = np.random.RandomState(123)
    base_samples = []
    for _ in range(50):
        length = rng.randint(10, 100)
        base_samples.append(rng.randint(1, 1000, size=length).tolist())

    samples = list(base_samples)
    for s in base_samples[:10]:
        samples.append(list(s))

    for s in base_samples[10:20]:
        mutated = list(s)
        if len(mutated) > 5:
            mutated[-1] = 9999
        samples.append(mutated)

    return samples
