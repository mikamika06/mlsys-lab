from ggufkit import container, plan


def assert_loadable(blob):
    """Raise AssertionError if this container is not safe to load.

    The message has to name the tensor at fault: an operator reading the log
    should know which tensor to look at without opening the file. A clean
    container must pass silently.
    """
    raise NotImplementedError
