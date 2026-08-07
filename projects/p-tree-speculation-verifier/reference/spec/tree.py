import numpy as np


class CandidateTree:
    def __init__(self, root_tokens):
        self.nodes = {}
        for i, tok in enumerate(root_tokens):
            self.nodes[i] = {"token": tok, "parent": -1, "children": [], "prob": 1.0}
        self.next_id = len(root_tokens)

    def add_child(self, parent_id, token_id, prob):
        nid = self.next_id
        self.next_id += 1
        self.nodes[nid] = {"token": token_id, "parent": parent_id, "children": [], "prob": prob}
        self.nodes[parent_id]["children"].append(nid)
        return nid

    def get_paths(self):
        paths = []
        def dfs(curr, path):
            p = path + [self.nodes[curr]["token"]]
            if not self.nodes[curr]["children"]:
                paths.append(p)
            else:
                for c in self.nodes[curr]["children"]:
                    dfs(c, p)
        roots = [k for k, v in self.nodes.items() if v["parent"] == -1]
        for r in roots:
            dfs(r, [])
        return paths


def build_tree_from_logits(draft_logits, max_depth, max_width):
    tree = CandidateTree([0])
    curr_level = [0]
    for d in range(max_depth):
        next_level = []
        for p in curr_level:
            logits = draft_logits[d]
            probs = np.exp(logits - np.max(logits))
            probs /= np.sum(probs)
            top_indices = np.argsort(probs)[-max_width:]
            for idx in top_indices:
                cid = tree.add_child(p, int(idx), float(probs[idx]))
                next_level.append(cid)
        curr_level = next_level
    return tree
