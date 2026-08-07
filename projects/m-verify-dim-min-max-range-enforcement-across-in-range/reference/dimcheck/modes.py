class DimMode:
    AUTO = "auto"
    DYNAMIC = "dynamic"
    EXPLICIT = "explicit"


def compare_modes(module, ambiguous_input):
    return {
        DimMode.AUTO: module(ambiguous_input, DimMode.AUTO),
        DimMode.DYNAMIC: module(ambiguous_input, DimMode.DYNAMIC),
        DimMode.EXPLICIT: module(ambiguous_input, DimMode.EXPLICIT)
    }
