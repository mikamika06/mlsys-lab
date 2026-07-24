def collect_yield_from(gen):
    """Drive gen to completion.

    Return (values, return_value) where:
      values:       list of all yielded values in order
      return_value: the StopIteration value from gen (None if unset)
    """
    values = []
    try:
        while True:
            values.append(next(gen))
    except StopIteration as exc:
        return values, exc.value
