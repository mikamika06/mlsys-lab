def select_longest_accepted_path(parents, accepted_mask):
    n = len(parents)
    best_len = -1
    best_path = []
    
    def dfs(curr, current_path):
        nonlocal best_len, best_path
        if curr != -1 and not accepted_mask[curr]:
            return
        
        path_so_far = current_path + ([curr] if curr != -1 else [])
        
        children = [i for i, p in enumerate(parents) if p == curr]
        if not children:
            if len(path_so_far) > best_len:
                best_len = len(path_so_far)
                best_path = list(path_so_far)
        else:
            leaf_reached = True
            for child in children:
                if accepted_mask[child]:
                    leaf_reached = False
                    dfs(child, path_so_far)
            if leaf_reached:
                if len(path_so_far) > best_len:
                    best_len = len(path_so_far)
                    best_path = list(path_so_far)

    dfs(-1, [])
    return best_path
