def grade(sol, fx) -> dict:
    import numpy as np

    def reference(weights):
        if len(weights) == 0 or len(weights[0]) % 4 != 0:
            raise ValueError("last dimension must be a multiple of 4")

        mask = []
        for row in weights:
            row_mask = []
            for i in range(0, len(row), 4):
                block = row[i:i + 4]
                abs_block = [abs(val) for val in block]

                max1_idx = 0
                for j in range(1, 4):
                    if abs_block[j] > abs_block[max1_idx]:
                        max1_idx = j

                max2_idx = -1
                for j in range(4):
                    if j == max1_idx:
                        continue
                    if max2_idx == -1 or abs_block[j] > abs_block[max2_idx]:
                        max2_idx = j

                block_mask = [False, False, False, False]
                block_mask[max1_idx] = True
                block_mask[max2_idx] = True
                row_mask.extend(block_mask)

            mask.append(row_mask)

        return mask

    rng = np.random.default_rng(42)

    test_cases = [
        [[0.1, -3.5, 2.0, 0.0, 1.2, 4.8, -0.7, 0.9],
         [5.0, -1.1, 0.3, 2.2, -4.4, 0.6, 3.3, -2.2]],
        rng.normal(size=(4, 12)).tolist(),
        rng.normal(size=(1, 4)).tolist(),
        rng.uniform(-10, 10, size=(10, 16)).tolist(),
    ]

    exact_matches = 0
    total_tests = len(test_cases)
    max_rel_err = 0.0

    for weights in test_cases:
        ref_mask = reference(weights)
        try:
            user_mask = sol.magnitude_optimal_2to4_mask(weights)
        except Exception:
            continue

        if not isinstance(user_mask, list) or not all(isinstance(r, list) for r in user_mask):
            continue

        if len(user_mask) != len(weights):
            continue

        shape_ok = True
        for r_user, r_weights in zip(user_mask, weights):
            if len(r_user) != len(r_weights):
                shape_ok = False
                break
        if not shape_ok:
            continue

        valid_pattern = True
        for row in user_mask:
            for i in range(0, len(row), 4):
                block = row[i:i + 4]
                if sum(1 for v in block if bool(v)) != 2:
                    valid_pattern = False
                    break
            if not valid_pattern:
                break

        if not valid_pattern:
            continue

        ref_sum = sum(
            abs(w) for r_w, r_m in zip(weights, ref_mask) for w, m in zip(r_w, r_m) if m
        )
        user_sum = sum(
            abs(w) for r_w, r_m in zip(weights, user_mask) for w, m in zip(r_w, r_m) if m
        )

        rel_err = abs(user_sum - ref_sum) / max(1e-12, abs(ref_sum))
        max_rel_err = max(max_rel_err, rel_err)

        if user_mask == ref_mask:
            exact_matches += 1

    exact_match = 1 if exact_matches == total_tests else 0
    sum_optimal = 1 if max_rel_err <= 1e-9 else 0

    return {
        "exact_match": exact_match,
        "sum_optimal": sum_optimal,
    }
