"""Regression tests for failure triage logic."""

import sys

sys.path.insert(0, ".")
from triage.classifier import triage_log_batch
from triage.isolate import isolate_root_cause


def test_tokenizer_damage_detection():
    sample = {
        "unk_token_ratio": 0.25,
        "bos_eos_missing": True,
        "id_out_of_bounds": False,
        "engine_panic": False,
        "buffer_overflow": False,
        "context_index_error": False,
        "ppl_spike": 2.0,
        "has_nan_inf": False,
        "logit_kl_divergence": 0.1,
    }
    assert isolate_root_cause(sample) == "tokenizer_damage"


def test_quantization_damage_detection():
    sample = {
        "unk_token_ratio": 0.01,
        "bos_eos_missing": False,
        "id_out_of_bounds": False,
        "engine_panic": False,
        "buffer_overflow": False,
        "context_index_error": False,
        "ppl_spike": 120.0,
        "has_nan_inf": True,
        "logit_kl_divergence": 4.5,
    }
    assert isolate_root_cause(sample) == "quantization_damage"


def test_engine_failure_detection():
    sample = {
        "unk_token_ratio": 0.0,
        "bos_eos_missing": False,
        "id_out_of_bounds": False,
        "engine_panic": True,
        "buffer_overflow": False,
        "context_index_error": True,
        "ppl_spike": 0.0,
        "has_nan_inf": False,
        "logit_kl_divergence": 0.0,
    }
    assert isolate_root_cause(sample) == "engine_failure"


def test_batch_triage():
    batch = [
        {"engine_panic": True},
        {"unk_token_ratio": 0.3},
        {"ppl_spike": 80.0},
    ]
    res = triage_log_batch(batch)
    assert res == ["engine_failure", "tokenizer_damage", "quantization_damage"]
