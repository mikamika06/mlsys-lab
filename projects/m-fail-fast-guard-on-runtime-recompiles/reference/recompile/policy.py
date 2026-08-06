def lookup_policy(table, state):
    return table.get(state, "default")
