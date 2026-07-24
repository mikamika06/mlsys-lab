def choose_budget_min_tail_itl(workload, candidate_budgets, ttft_cap):
    """Pick the token budget that minimizes tail (worst-case) inter-token
    latency, subject to every request meeting a TTFT cap.

    workload: list of dicts, each {"id": int, "arrival": int,
        "prompt_len": int, "decode_len": int}. `id` is unique;
        `prompt_len` and `decode_len` are >= 1.
    candidate_budgets: list of positive ints -- the per-iteration token
        budgets to evaluate.
    ttft_cap: int -- the max allowed worst-case time-to-first-token, in the
        same time units the simulation uses.

    Scheduler model (per candidate budget `b`), simulated over discrete
    iterations with a running wall-clock time `T` starting at 0:
      - Requests prefill strictly FCFS, ordered by (arrival, id); only one
        request prefills at a time.
      - Each iteration, every currently-decoding request unconditionally
        receives exactly 1 token (there are `num_decoders` of them this
        iteration).
      - Whatever budget remains, `max(b - num_decoders, 0)`, is spent
        continuing the current prefill target: `chunk = min(leftover,
        remaining_prompt)`.
      - The iteration's duration is `num_decoders + chunk` (the number of
        tokens actually processed); `T` advances by that amount, and every
        token produced this iteration (all `num_decoders` of them) is
        timestamped at the new `T`.
      - If the prefill target's prompt is fully consumed this iteration,
        it becomes a decoding request starting the *next* iteration.
      - If nothing can be done this iteration (no active decoders and no
        arrived prefill target), fast-forward `T` to the next arrival.

    For each request: TTFT = timestamp of its first produced token minus
    its arrival; ITL = the largest gap between consecutive produced-token
    timestamps (0 if it produces fewer than 2 tokens).

    For each candidate budget, compute (max_ttft, max_itl) across the
    whole workload via this simulation. Among candidates with
    max_ttft <= ttft_cap, return the one with the smallest max_itl (ties
    broken by the smaller budget value). If no candidate satisfies the
    cap, return -1.
    """
    raise NotImplementedError('your code here')
