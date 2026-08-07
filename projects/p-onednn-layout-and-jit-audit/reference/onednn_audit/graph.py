class OpNode:
    def __init__(self, op_id, prim_kind, in_layout, out_layout, exec_time_ms):
        self.op_id = op_id
        self.prim_kind = prim_kind
        self.in_layout = in_layout
        self.out_layout = out_layout
        self.exec_time_ms = float(exec_time_ms)

    def is_reorder(self):
        return self.prim_kind == "reorder"


class ExecutionGraph:
    def __init__(self, nodes):
        self.nodes = list(nodes)

    def find_layout_transitions(self):
        transitions = []
        for i in range(len(self.nodes) - 1):
            curr_node = self.nodes[i]
            next_node = self.nodes[i + 1]
            if curr_node.out_layout != next_node.in_layout:
                transitions.append((curr_node.op_id, next_node.op_id, curr_node.out_layout, next_node.in_layout))
        return transitions

    def find_redundant_reorders(self):
        redundant = []
        for i in range(1, len(self.nodes) - 1):
            curr_node = self.nodes[i]
            if curr_node.is_reorder():
                prev_node = self.nodes[i - 1]
                next_node = self.nodes[i + 1]
                if prev_node.out_layout == next_node.in_layout:
                    redundant.append(curr_node.op_id)
        return redundant

    def total_reorder_time_ms(self):
        return sum(node.exec_time_ms for node in self.nodes if node.is_reorder())

    def total_execution_time_ms(self):
        return sum(node.exec_time_ms for node in self.nodes)
