def check_stop_sequences(template, stop_seq, sample_text):
    if not stop_seq:
        return False
    return stop_seq in sample_text
