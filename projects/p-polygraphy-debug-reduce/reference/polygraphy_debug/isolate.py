class GraphIsolator:
    def __init__(self, execution_graph):
        self.graph = execution_graph

    def extract_subgraph(self, target_node_id):
        deps = self.graph.get_dependencies(target_node_id)
        sub_nodes = sorted(list(deps | {target_node_id}))
        return {
            "target": target_node_id,
            "subgraph_nodes": sub_nodes,
            "node_spec": self.graph.get_node_spec(target_node_id)
        }
