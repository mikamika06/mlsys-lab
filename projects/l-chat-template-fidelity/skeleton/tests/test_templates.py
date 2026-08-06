def run_suite(render, funcs, fixtures):
    """Replay every recorded rendering in the fixture directory.

    Returns (passed, failed). `fixtures` is the templates fixture directory:
    it holds inputs.json, the two renderings files, and the semantics/
    templates; the ollama templates sit in a sibling directory.

    inputs.json is plain JSON. Two values in it print as compact JSON rather
    than as a Python container, and rebuilding that is part of the job.
    """
    raise NotImplementedError
