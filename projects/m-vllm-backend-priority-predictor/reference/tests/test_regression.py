import sys

sys.path.insert(0, ".")
from vllm_pred.parser import parse_log
from vllm_pred.predictor import predict_backend
from vllm_pred.rejection import get_rejection_reason


def test_parse_and_predict():
    log = "INFO 00-00 [attention.py] Evaluated backends: ['FLASH_ATTN', 'XFORMERS']\nINFO 00-00 [attention.py] Backend FLASH_ATTN rejected because sm too low\nINFO 00-00 [attention.py] Selected attention backend: XFORMERS"
    res = parse_log(log)
    assert res["selected"] == "XFORMERS"
    assert get_rejection_reason("FLASH_ATTN", res["rejections"]) == "sm too low"
    assert predict_backend(res["evaluated"], res["rejections"]) == "XFORMERS"
