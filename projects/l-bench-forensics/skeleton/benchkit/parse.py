RESULT_FIELDS = {"avg_ns", "stddev_ns", "avg_ts", "stddev_ts", "samples_ns",
                 "samples_ts", "test_time"}
BUILD_FIELDS = {"build_commit", "build_number"}


def load(path):
    """One llama-bench JSON file. Tag each row with its file and index."""
    raise NotImplementedError


def load_all(paths):
    raise NotImplementedError


def kind(row):
    """"prefill", "decode" or "mixed", from n_prompt and n_gen."""
    raise NotImplementedError


def tokens(row):
    raise NotImplementedError


def derive(row):
    """Everything worth having per row: kind, tokens, avg_seconds,
    tokens_per_second, ms_per_token, depth, ubatch, batch, model, samples_ts,
    reps, source, row.

    Compute throughput from avg_ns rather than copying avg_ts. They agree here,
    and the point is that you can say so.
    """
    raise NotImplementedError


def config(row):
    """The knobs a run was set with. Results, timestamps and build identifiers
    are not configuration, and neither is anything that arrived as a list."""
    raise NotImplementedError
