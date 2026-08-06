import ref
from gguf_triage.classifier import classify_log, get_fixing_command
from gguf_triage.repair import repair_architecture

def test_classification():
    for log, cause in ref.LOGS:
        assert classify_log(log) == cause

def test_fixing_commands():
    for _, cause in ref.LOGS:
        cmd = get_fixing_command(cause)
        assert isinstance(cmd, str)
        assert len(cmd) > 0

def test_repair_function():
    sample = ref.make_sample_gguf()
    repaired = repair_architecture(sample, "llama")
    assert len(repaired) >= len(sample)
