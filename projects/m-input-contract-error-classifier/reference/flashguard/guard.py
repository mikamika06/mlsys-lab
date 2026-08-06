from typing import Dict, Any, Tuple
import numpy as np
from .classifier import ErrorCategory, InputContractError, classify_input_error


def guard_and_realign(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    alignment_bytes: int = 16,
    max_head_dim: int = 256,
    auto_realign: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    err = classify_input_error(q, k, v, alignment_bytes, max_head_dim)

    if err == ErrorCategory.NONE:
        return q, k, v, {"realigned": False, "category": err.value}

    if err == ErrorCategory.NON_CONTIGUOUS and auto_realign:
        q_c = np.ascontiguousarray(q) if not q.flags["C_CONTIGUOUS"] else q
        k_c = np.ascontiguousarray(k) if not k.flags["C_CONTIGUOUS"] else k
        v_c = np.ascontiguousarray(v) if not v.flags["C_CONTIGUOUS"] else v

        post_err = classify_input_error(q_c, k_c, v_c, alignment_bytes, max_head_dim)
        if post_err == ErrorCategory.NONE:
            return q_c, k_c, v_c, {"realigned": True, "category": ErrorCategory.NONE.value}

    raise InputContractError(err, f"Input contract violation: {err.value}")


class GuardManager:
    def __init__(self, alignment_bytes: int = 16, max_head_dim: int = 256):
        self.alignment_bytes = alignment_bytes
        self.max_head_dim = max_head_dim
        self.stats = {"validated": 0, "realigned": 0, "rejected": 0}

    def process(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        auto_realign: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            q_out, k_out, v_out, info = guard_and_realign(
                q,
                k,
                v,
                alignment_bytes=self.alignment_bytes,
                max_head_dim=self.max_head_dim,
                auto_realign=auto_realign,
            )
            self.stats["validated"] += 1
            if info.get("realigned", False):
                self.stats["realigned"] += 1
            return q_out, k_out, v_out
        except InputContractError:
            self.stats["rejected"] += 1
            raise
