def slot_occupancy_trajectory(reqs: list[tuple[int, int]], N: int) -> list[list[int]]:
    """Simulate a llama.cpp-server-style fixed-slot decode scheduler
    (`--parallel N` / `-np N`): N physical decode slots, each holding one
    request from admission until it finishes generating.

    reqs : list of (arrival_step, gen_len) pairs. Request `i`'s id is its
           index into this list.
    N    : number of fixed decode slots.

    At each discrete step t (starting at 0):
      1. Free any slot whose occupant already finished (generated
         `gen_len` tokens as of the end of the previous step).
      2. Move every request with `arrival_step <= t` that hasn't been
         admitted yet into the waiting queue (FIFO order: by
         `arrival_step`, ties broken by request id).
      3. Fill free slots from the front of the waiting queue, one
         request per slot, as many as fit.
      4. Every occupied slot decodes exactly one token this step
         (progress += 1). A slot that reaches `gen_len` this step stays
         visibly occupied THIS step's snapshot; it is freed at step 1 of
         the NEXT step.
      5. Record this step's occupancy: a length-N list of request ids
         (or -1 for an idle slot).

    The simulation ends the step after the last request finishes.

    Returns a list of steps, each a length-N list of ints.
    """
    n_reqs = len(reqs)
    admission_order = sorted(range(n_reqs), key=lambda i: (reqs[i][0], i))

    slots: list[dict | None] = [None] * N
    waiting: list[int] = []
    admit_ptr = 0
    finished = 0

    trajectory: list[list[int]] = []
    t = 0
    # generous safety bound; the real loop always terminates on its own
    safety = sum(g for _, g in reqs) + n_reqs + N + 10
    while t < safety:
        # 1. free finished slots
        for s in range(N):
            occ = slots[s]
            if occ is not None and occ["progress"] >= reqs[occ["req"]][1]:
                slots[s] = None
                finished += 1

        # done: nothing left to admit, nothing waiting, nothing busy
        if admit_ptr >= n_reqs and not waiting and all(s is None for s in slots):
            break

        # 2. move newly arrived requests into the waiting queue
        while admit_ptr < n_reqs and reqs[admission_order[admit_ptr]][0] <= t:
            waiting.append(admission_order[admit_ptr])
            admit_ptr += 1

        # 3. fill free slots from the waiting queue, FIFO
        for s in range(N):
            if slots[s] is None and waiting:
                slots[s] = {"req": waiting.pop(0), "progress": 0}

        # 4. decode one token per busy slot
        for s in range(N):
            if slots[s] is not None:
                slots[s]["progress"] += 1

        # 5. record occupancy
        trajectory.append([slots[s]["req"] if slots[s] is not None else -1 for s in range(N)])
        t += 1

    return trajectory
