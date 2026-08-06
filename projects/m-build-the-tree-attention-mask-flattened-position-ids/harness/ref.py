import numpy as np

def generate_trees(n):
    rng = np.random.RandomState(42)
    trees = []
    for _ in range(n):
        size = rng.randint(5, 20)
        parents = [-1]
        for i in range(1, size):
            parents.append(rng.randint(0, i))
        trees.append(parents)
    return trees

def build_tree_mask_and_positions(parents, root_pos):
    n = len(parents)
    mask = np.zeros((n, n), dtype=np.int32)
    pos = np.zeros(n, dtype=np.int32)

    for i in range(n):
        pos[i] = root_pos if i == 0 else pos[parents[i]] + 1
        curr = i
        while curr != -1:
            mask[i, curr] = 1
            curr = parents[curr]

    return mask, pos

def select_longest_path(parents, accepted_nodes):
    n = len(parents)
    dp = np.zeros(n, dtype=np.int32)
    
    if accepted_nodes[0]:
        dp[0] = 1
        
    for i in range(1, n):
        if accepted_nodes[i] and dp[parents[i]] > 0:
            dp[i] = dp[parents[i]] + 1
            
    if np.max(dp) == 0:
        return []
        
    best_end = int(np.argmax(dp))
    path = []
    curr = best_end
    
    while curr != -1 and dp[curr] > 0:
        path.append(curr)
        curr = parents[curr]
        
    return path[::-1]
