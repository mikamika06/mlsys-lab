class RadixEvictionManager:
    def __init__(self, capacity):
        self.capacity = capacity

    def evict(self, tree_root, current_usage):
        leaves = []
        def collect_leaves(node):
            if not node.children:
                if node.ref_count == 0 and node != tree_root:
                    leaves.append(node)
            for child in node.children.values():
                collect_leaves(child)

        collect_leaves(tree_root)
        leaves.sort(key=lambda n: n.last_access)

        evicted = 0
        for leaf in leaves:
            if current_usage <= self.capacity:
                break
            def remove_node(curr, target):
                for k, child in list(curr.children.items()):
                    if child == target:
                        del curr.children[k]
                        return True
                    if remove_node(child, target):
                        if not child.children and child.ref_count == 0 and child.value is not None:
                            del curr.children[k]
                        return True
                return False
            if remove_node(tree_root, leaf):
                evicted += 1
                current_usage -= 1
        return evicted
