def make_truthful_type(name, accepted_type):
    # TODO: this creates a normal class, so isinstance only checks inheritance
    # and ignores the intended accepted_type relationship.
    return type(name, (), {})
