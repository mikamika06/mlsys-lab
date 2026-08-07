class Node:
    def __init__(self, name, op_type, inputs, outputs):
        raise NotImplementedError

class Graph:
    def __init__(self, nodes):
        raise NotImplementedError

    def execute(self, inputs_dict):
        raise NotImplementedError
