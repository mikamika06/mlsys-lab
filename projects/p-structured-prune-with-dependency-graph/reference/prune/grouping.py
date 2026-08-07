class Group:
    """Group of coupled tensor dimensions pruned together."""

    def __init__(self, group_id):
        self.group_id = group_id
        self.items = []

    def add_item(self, node, axis):
        if (node, axis) not in self.items:
            self.items.append((node, axis))


class GroupFinder:
    """Disjoint-set finder for identifying coupled pruning groups."""

    def __init__(self, dep_graph):
        self.graph = dep_graph

    def get_pruning_groups(self):
        parent = {}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        for name, node in self.graph.nodes.items():
            if node.layer.layer_type in ["linear", "conv2d"]:
                parent[(node, 0)] = (node, 0)
                parent[(node, 1)] = (node, 1)
            elif node.layer.layer_type in ["batchnorm", "add"]:
                parent[(node, 0)] = (node, 0)

        for dep in self.graph.dependencies:
            src_item = (dep.src_node, dep.src_axis)
            dst_item = (dep.dst_node, dep.dst_axis)
            if src_item in parent and dst_item in parent:
                union(src_item, dst_item)

        for name, node in self.graph.nodes.items():
            if node.layer.layer_type == "add" and len(node.inputs) >= 2:
                first_in = (self.graph.nodes[node.inputs[0]], 0)
                for other_in_name in node.inputs[1:]:
                    other_in = (self.graph.nodes[other_in_name], 0)
                    if first_in in parent and other_in in parent:
                        union(first_in, other_in)

        groups_dict = {}
        for item in parent:
            root = find(item)
            if root not in groups_dict:
                groups_dict[root] = Group(len(groups_dict))
            groups_dict[root].add_item(item[0], item[1])

        return list(groups_dict.values())
