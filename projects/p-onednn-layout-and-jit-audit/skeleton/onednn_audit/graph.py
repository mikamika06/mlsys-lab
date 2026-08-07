class OpNode:
    def __init__(self, op_id, prim_kind, in_layout, out_layout, exec_time_ms):
        raise NotImplementedError


class ExecutionGraph:
    def __init__(self, nodes):
        raise NotImplementedError

    def find_layout_transitions(self):
        raise NotImplementedError

    def find_redundant_reorders(self):
        raise NotImplementedError

    def total_reorder_time_ms(self):
        raise NotImplementedError

    def total_execution_time_ms(self):
        raise NotImplementedError
