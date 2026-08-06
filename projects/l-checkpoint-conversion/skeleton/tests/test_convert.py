def audit(tensors, meta, prefix, experts=0):
    """Reasons not to run this conversion, as a list of strings.

    Empty on a clean checkpoint. The real fixtures are not clean: one grows
    several times over on the way out because the source is quantised, and one
    carries fused expert tensors that go nowhere without an expert count.
    """
    raise NotImplementedError
