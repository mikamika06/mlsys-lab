def graph_launch_step_time(L: float, N: int, C: float) -> list[float]:
    """
    L: per-kernel launch overhead (time units).
    N: number of kernels issued per step.
    C: total GPU compute time per step (time actually spent executing on
        the device, independent of how many separate kernel launches it
        took to issue that work).

    Eager execution pays the launch overhead L once PER kernel; a single
    captured CUDA graph replaces all N per-kernel launches with ONE graph
    launch, paying L only once for the whole step.

    Returns [eager_time, graph_time, fraction_removed]:
      eager_time = N * L + C
      graph_time = L + C
      fraction_removed = (eager_time - graph_time) / eager_time
    """
    raise NotImplementedError('your code here')
