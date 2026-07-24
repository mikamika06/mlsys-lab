def rank_by_mlp() -> list:
    """
    Return pattern names ranked from lowest MLP to highest MLP.

    MLP reasoning:
    - pointer_chase: MLP=1 (each access depends on previous value)
    - sequential: moderate MLP (HW prefetcher generates parallel misses)
    - strided: higher MLP (multiple stride streams can be in-flight)
    - scatter_gather: highest MLP (many independent random streams)
    """
    return ["pointer_chase", "sequential", "strided", "scatter_gather"]
