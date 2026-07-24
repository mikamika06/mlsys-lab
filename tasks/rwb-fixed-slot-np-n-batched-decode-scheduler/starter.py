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

    The simulation ends the step after the last request finishes (no
    trailing all-idle step is recorded).

    Returns a list of steps, each a length-N list of ints.
    """
    raise NotImplementedError('your code here')
