def backward_pass(tape, values, n_inputs):
    """Run reverse-mode autodiff over a Wengert tape.

    `tape` is a list of `(op, args)` pairs. `values` is the full forward
    trace: `values[0:n_inputs]` are the leaf inputs and `values[n_inputs+k]`
    is the value produced by `tape[k]`, computed from earlier entries only
    (so the tape is already in topological order). `values[-1]` is the
    scalar output.

    Returns a list/array of length `n_inputs`: d(values[-1]) / d(values[i])
    for each input i.
    """
    raise NotImplementedError('your code here')
