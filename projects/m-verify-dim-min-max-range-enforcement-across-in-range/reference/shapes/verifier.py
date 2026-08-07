class DimType:
    AUTO = "AUTO"
    DYNAMIC = "DYNAMIC"
    EXPLICIT = "EXPLICIT"


class ConstraintViolationError(Exception):
    def __init__(self, name, val, min_val, max_val):
        self.name = name
        self.val = val
        self.min_val = min_val
        self.max_val = max_val
        super().__init__(self.suggested_fix())

    def suggested_fix(self):
        return f"Dimension '{self.name}' got size {self.val} which is outside range [{self.min_val}, {self.max_val}]. Increase max_val if this batch size is expected."


class Dim:
    def __init__(self, name, min_val=1, max_val=2147483647, dim_type=DimType.EXPLICIT):
        self.name = name
        self.min_val = min_val
        self.max_val = max_val
        self.dim_type = dim_type

    @classmethod
    def auto(cls, name):
        return cls(name, dim_type=DimType.AUTO)

    @classmethod
    def dynamic(cls, name):
        return cls(name, dim_type=DimType.DYNAMIC)


def verify_range(dim, batch_size):
    if not (dim.min_val <= batch_size <= dim.max_val):
        raise ConstraintViolationError(dim.name, batch_size, dim.min_val, dim.max_val)
    return True


def resolve_module_signature(dim_specs, observed_shapes):
    resolved = []
    for i, dim in enumerate(dim_specs):
        sizes = [shape[i] for shape in observed_shapes]
        if dim.dim_type == DimType.AUTO:
            if len(set(sizes)) > 1:
                raise ValueError(f"AUTO dimension {dim.name} saw varying sizes: {sizes}")
            resolved.append(Dim(dim.name, sizes[0], sizes[0], DimType.EXPLICIT))
        elif dim.dim_type == DimType.DYNAMIC:
            resolved.append(Dim(dim.name, min(sizes), max(sizes), DimType.EXPLICIT))
        else:
            for s in sizes:
                verify_range(dim, s)
            resolved.append(Dim(dim.name, dim.min_val, dim.max_val, DimType.EXPLICIT))
    return resolved
