import numpy as np

def unpack_cusparselt_metadata(metadata: np.ndarray, shape: tuple[int,int]) -> np.ndarray:
    """
    Reconstruct a dense boolean mask from cuSPARSELt‑style packed metadata.

    Parameters
    ----------
    metadata : np.ndarray of dtype uint8
        Packed two‑bit codes; each byte contains four consecutive codes.
    shape : tuple(int, int)
        Desired output dimensions (rows, cols).

    Returns
    -------
    np.ndarray of dtype bool
        Dense mask of the requested shape.
    """
    total = shape[0] * shape[1]
    bits_needed = total

    # Extract all two‑bit codes from the metadata
    codes = []
    for byte in metadata:
        for k in range(4):
            code = (byte >> (k * 2)) & 0x03
            codes.append(code != 0)

    mask_flat = np.array(codes[:bits_needed], dtype=bool)
    return mask_flat.reshape(shape)
