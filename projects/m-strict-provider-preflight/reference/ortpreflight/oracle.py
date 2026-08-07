def parse_version(v_str):
    return tuple(int(x) for x in v_str.split("."))


def check_cuda_cudnn_compat(ort_version, cuda_version, cudnn_version):
    ort_tuple = parse_version(ort_version)
    cuda_tuple = parse_version(cuda_version)
    cudnn_tuple = parse_version(cudnn_version)

    ort_major_minor = ort_tuple[:2]

    supported_cuda = {
        (1, 15): [(11, 8), (12, 0)],
        (1, 16): [(11, 8), (12, 2)],
        (1, 17): [(11, 8), (12, 2)],
        (1, 18): [(11, 8), (12, 4)],
    }

    if ort_major_minor not in supported_cuda:
        return False, f"Unsupported ORT version {ort_version}"

    cuda_mm = cuda_tuple[:2]
    if cuda_mm not in supported_cuda[ort_major_minor]:
        return False, f"Unsupported CUDA {cuda_version} for ORT {ort_version}"

    if cuda_mm == (11, 8):
        if not ((8, 2) <= cudnn_tuple[:2] < (9, 0)):
            return False, f"Incompatible cuDNN {cudnn_version} for CUDA 11.8"
    elif cuda_mm == (12, 0):
        if not ((8, 8) <= cudnn_tuple[:2] < (9, 0)):
            return False, f"Incompatible cuDNN {cudnn_version} for CUDA 12.0"
    elif cuda_mm == (12, 2):
        if ort_major_minor == (1, 16):
            if not ((8, 8) <= cudnn_tuple[:2] < (9, 0)):
                return False, f"Incompatible cuDNN {cudnn_version} for CUDA 12.2"
        else:
            if not ((8, 9) <= cudnn_tuple[:2] < (9, 0)):
                return False, f"Incompatible cuDNN {cudnn_version} for CUDA 12.2"
    elif cuda_mm == (12, 4):
        if not ((9, 0) <= cudnn_tuple[:2] < (10, 0)):
            return False, f"Incompatible cuDNN {cudnn_version} for CUDA 12.4"

    return True, "Compatible"
