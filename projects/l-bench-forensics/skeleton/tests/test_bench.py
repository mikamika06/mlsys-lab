def audit(rows):
    """Everything about this measurement set that should stop a decision.

    A list of strings, empty when the data is clean. Each string names the file
    and row it is about. The real fixture contains at least one row that a
    cache-behaviour story cannot explain; clean data must produce nothing.
    """
    raise NotImplementedError
