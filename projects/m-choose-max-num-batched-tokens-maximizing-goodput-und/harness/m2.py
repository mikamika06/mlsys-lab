import ref
from chunking.correctness import verify_chunked_prefill_logits
from chunking.translate import translate_vllm_to_sglang


def check(workdir):
    out = {"logits_matched": 0.0}
    full = [[0.5, 0.3, 0.2], [0.1, 0.8, 0.1]]
    chunked = [[0.5, 0.3, 0.2], [0.1, 0.8, 0.1]]
    
    res_correct = verify_chunked_prefill_logits(full, chunked)
    
    cfg = {"max_num_batched_tokens": -1, "enable_chunked_prefill": True}
    translated = translate_vllm_to_sglang(cfg)
    
    if res_correct is True and translated.get("chunked_prefill_size") == 0:
        out["logits_matched"] = 1.0
    else:
        out["_note"] = f"verification failed or translation incorrect: res={res_correct}, translated={translated}"
    return out
