def predict_perm(regkey, groups, root):
    if root:
        return False
    if regkey == 1 and "nsight" in groups:
        return False
    return True
