def classify_breaks(snippets: list[str]) -> list[str]:
    # TODO: this naive implementation assumes all snippets are traceable,
    # which is incorrect for many cases.
    return ["traceable"] * len(snippets)
