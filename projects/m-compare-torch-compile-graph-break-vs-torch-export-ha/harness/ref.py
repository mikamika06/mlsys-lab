RECORDS = [
    {"has_data_dependent_branch": True, "graph_breaks": 4},
    {"has_data_dependent_branch": False, "graph_breaks": 0},
    {"has_data_dependent_branch": True, "graph_breaks": 5}
]


def get_expected_analysis():
    return [
        {"compile_graph_breaks": 4, "export_hard_error": True, "resolvable": False},
        {"compile_graph_breaks": 0, "export_hard_error": False, "resolvable": True},
        {"compile_graph_breaks": 5, "export_hard_error": True, "resolvable": False}
    ]


def get_valid_test_inputs():
    return True, 3.5, 4.2
