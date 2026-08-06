def count_graph_breaks(hooks, splitting):
    breaks = 0
    if hooks:
        breaks += 1
    if splitting:
        breaks += 2
    return breaks
