def validate_group_size(in_features, group_size):
    if in_features % group_size != 0:
        raise ValueError(f"in_features {in_features} is not divisible by group_size {group_size}")
    return True
