def build_histogram(tokens):
    histogram = {}
    for token in tokens:
        histogram[token] = histogram.get(token, 0) + 1
    return histogram
