import ref


def check(workdir):
    from gparser.parser import parse_graph
    from gparser.inspector import inspect_shapes, inspect_dtypes, inspect_io
    from gparser.optimizer import optimize_graph

    out = {"shapes_match": 0.0, "dtypes_match": 0.0, "io_match": 0.0, "opt_match": 0.0}
    prog = ref.PROGRAMS[0]
    parsed_want = ref.parse_graph(prog)
    parsed_got = parse_graph(prog)
    if inspect_shapes(parsed_got) == ref.inspect_shapes(parsed_want):
        out["shapes_match"] = 1.0
    if inspect_dtypes(parsed_got) == ref.inspect_dtypes(parsed_want):
        out["dtypes_match"] = 1.0
    if inspect_io(parsed_got) == ref.inspect_io(parsed_want):
        out["io_match"] = 1.0
    if optimize_graph(parsed_got) == ref.optimize_graph(parsed_want):
        out["opt_match"] = 1.0
    return out
