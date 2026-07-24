import numpy as np


def _oracle(prompt_lens, C):
    N = len(prompt_lens)

    ttft_mono = [0] * N
    i = 0
    iters_mono = 0
    while i < N:
        iters_mono += 1
        if prompt_lens[i] > C:
            ttft_mono[i] = iters_mono
            i += 1
            continue
        budget = C
        while i < N and prompt_lens[i] <= budget:
            budget -= prompt_lens[i]
            ttft_mono[i] = iters_mono
            i += 1

    remaining = list(prompt_lens)
    ttft_chunked = [0] * N
    i = 0
    iters_chunked = 0
    while i < N:
        iters_chunked += 1
        budget = C
        while i < N and budget > 0:
            take = min(remaining[i], budget)
            remaining[i] -= take
            budget -= take
            if remaining[i] == 0:
                ttft_chunked[i] = iters_chunked
                i += 1
            else:
                break

    return {
        "iters_mono": iters_mono,
        "iters_chunked": iters_chunked,
        "ttft_mono": ttft_mono,
        "ttft_chunked": ttft_chunked,
    }


def _synthetic_cases():
    rng = np.random.default_rng(41)
    cases = []
    for _ in range(4):
        N = int(rng.integers(3, 15))
        prompt_lens = rng.integers(1, 60, size=N).tolist()
        C = int(rng.integers(8, 40))
        cases.append((prompt_lens, C))
    return cases


def grade(sol, fx) -> dict:
    fixture_case = (fx["prompt_lens"].tolist(), int(fx["budget"]))
    cases = [fixture_case] + _synthetic_cases()

    total = 0
    correct = 0
    for prompt_lens, C in cases:
        ref = _oracle(prompt_lens, C)
        N = len(prompt_lens)
        try:
            got = sol.compare_chunked_vs_monolithic(list(prompt_lens), C)
        except Exception:
            total += 2 + 2 * N
            continue

        for key in ("iters_mono", "iters_chunked"):
            total += 1
            try:
                if int(got[key]) == ref[key]:
                    correct += 1
            except Exception:
                pass

        for key in ("ttft_mono", "ttft_chunked"):
            total += N
            try:
                got_list = [int(x) for x in got[key]]
                if len(got_list) == N and got_list == ref[key]:
                    correct += N
                elif len(got_list) == N:
                    correct += sum(1 for a, b in zip(got_list, ref[key]) if a == b)
            except Exception:
                pass

    return {"exact_match": (correct / total) if total else 0.0}
