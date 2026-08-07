class Node:
    """Dependency graph node representing a layer operator."""

    def __init__(self, name, layer):
        self.name = name
        self.layer = layer
        self.inputs = []
        self.outputs = []


class Dependency:
    """Directed dependency link between two graph nodes."""

    def __init__(self, src_node, dst_node, src_axis, dst_axis):
        self.src_node = src_node
        self.dst_node = dst_node
        self.src_axis = src_axis
        self.dst_axis = dst_axis


class DependencyGraph:
    """Graph structure for tracking parameter tensor dependencies."""

    def __init__(self):
        self.nodes = {}
        self.dependencies = []

    def build(self, model):
        for name, layer in model.layers.items():
            self.nodes[name] = Node(name, layer)
        for src_name, dst_name in model.connections:
            src_node = self.nodes[src_name]
            dst_node = self.nodes[dst_name]
            src_node.outputs.append(dst_name)
            dst_node.inputs.append(src_name)

            src_axis = 0
            if dst_node.layer.layer_type in ["linear", "conv2d"]:
                dst_axis = 1
            else:
                dst_axis = 0
            self.dependencies.append(Dependency(src_node, dst_node, src_axis, dst_axis))
