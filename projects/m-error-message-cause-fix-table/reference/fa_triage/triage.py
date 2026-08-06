ERROR_TABLE = [
    {
        "pattern": "cudaErrorInvalidConfiguration",
        "cause": "Grid or block dimensions exceed device limits or stride alignment is invalid for flash attention execution.",
        "fix": "Ensure head dimension is a multiple of 32 and block size aligns with warp size constraints."
    },
    {
        "pattern": "cudaErrorIllegalAddress",
        "cause": "Page table or block table index out of bounds during PagedAttention memory lookup.",
        "fix": "Verify max sequence length and ensure physical page table allocations match context slots."
    },
    {
        "pattern": "cudaErrorAssert",
        "cause": "Internal kernel assertion triggered due to unsupported scale factor or head dimension.",
        "fix": "Check scaling factors and verify head dimension is supported by the installed flash attention kernel variant."
    }
]

def lookup_error(message: str) -> dict:
    for entry in ERROR_TABLE:
        if entry["pattern"] in message:
            return entry
    return {
        "pattern": "unknown",
        "cause": "Unrecognized kernel fault.",
        "fix": "Collect full stack trace and dump tensor shapes."
    }
