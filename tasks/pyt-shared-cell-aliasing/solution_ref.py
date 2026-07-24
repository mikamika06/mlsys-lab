def shared_cell_trace():
    def outer():
        value = 10

        def mutate():
            nonlocal value
            value += 3
            return value

        def observe():
            return value

        return mutate, observe

    mutate, observe = outer()
    return (
        observe(),
        mutate(),
        observe(),
        mutate(),
        observe(),
    )
