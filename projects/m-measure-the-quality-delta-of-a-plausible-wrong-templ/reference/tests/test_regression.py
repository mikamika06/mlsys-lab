import sys
sys.path.insert(0, ".")
from templater.stops import check_stop_sequences

def test_stop_sequence_termination():
    template = "USER: {prompt}\nASSISTANT:"
    stop_seq = "<|end|>"
    sample_without_stop = "Here is the response without token boundary"
    sample_with_stop = "Here is the response <|end|>"

    assert not check_stop_sequences(template, stop_seq, sample_without_stop)
    assert check_stop_sequences(template, stop_seq, sample_with_stop)

def test_empty_stop_sequence():
    template = "USER: {prompt}\nASSISTANT:"
    assert not check_stop_sequences(template, "", "Some text")
