def select_victim(root):
    candidates = []
    def dfs(node):
        if not node.children and node != root and node.ref_count <= 1:
            candidates.append(node)
        for child in node.children.values():
            dfs(child)
    dfs(root)
    if not candidates:
        return None
    return min(candidates, key=lambda n: len(n.key))
