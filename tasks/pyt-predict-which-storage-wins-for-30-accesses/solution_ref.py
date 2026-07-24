def predict_storage_wins(accesses, class_dict, instance_dict, descriptor_flags):
    result = []

    for name in accesses:
        kind = descriptor_flags.get(name)

        if kind == "data":
            result.append(0)
        elif name in instance_dict:
            result.append(1)
        elif kind == "nondata":
            result.append(2)
        elif name in class_dict:
            result.append(3)
        else:
            result.append(4)

    return result
