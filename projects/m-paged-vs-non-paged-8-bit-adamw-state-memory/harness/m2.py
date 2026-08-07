import sys
import numpy as np
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from qlora_mem.lora_merge import dequantize_4bit, merge_lora_into_base, quantize_to_4bit
    except Exception as e:
        return {"dequant_rel_err": 1.0, "merged_rel_err": 1.0, "requant_rel_err": 1.0, "_note": f"Import error: {e}"}

    qweights, scales, lora_A, lora_B, alpha, block_size = ref.generate_quant_fixture()

    ref_dequant = ref.dequantize_4bit(qweights, scales, block_size)
    got_dequant = dequantize_4bit(qweights, scales, block_size)
    dequant_err = float(np.linalg.norm(got_dequant - ref_dequant) / (np.linalg.norm(ref_dequant) + 1e-8))

    ref_merged = ref.merge_lora_into_base(qweights, scales, lora_A, lora_B, alpha, block_size)
    got_merged = merge_lora_into_base(qweights, scales, lora_A, lora_B, alpha, block_size)
    merged_err = float(np.linalg.norm(got_merged - ref_merged) / (np.linalg.norm(ref_merged) + 1e-8))

    got_q, got_s = quantize_to_4bit(ref_merged, block_size)
    reconstructed = ref.dequantize_4bit(got_q, got_s, block_size)
    requant_err = float(np.linalg.norm(reconstructed - ref_merged) / (np.linalg.norm(ref_merged) + 1e-8))

    return {
        "dequant_rel_err": dequant_err,
        "merged_rel_err": merged_err,
        "requant_rel_err": requant_err,
    }
