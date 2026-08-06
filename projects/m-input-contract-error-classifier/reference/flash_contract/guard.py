def check_contiguity(strides):
    if not strides:
        return False
    return strides[-1] == 1
