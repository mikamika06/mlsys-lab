def validate_guard(shape, bucket, guard_type):
    if guard_type == "shape":
        return shape <= bucket
    elif guard_type == "value":
        return shape == bucket
    elif guard_type == "id":
        return shape == bucket
    return False
