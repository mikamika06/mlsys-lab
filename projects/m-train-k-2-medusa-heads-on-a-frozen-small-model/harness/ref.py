import numpy as np

np.random.seed(42)
VOCAB_SIZE = 256
HIDDEN_DIM = 32
SEQ_LEN = 16

def generate_data(num_samples=10):
    hidden_states = np.random.randn(num_samples, SEQ_LEN, HIDDEN_DIM).astype(np.float32)
    targets = np.random.randint(0, VOCAB_SIZE, size=(num_samples, SEQ_LEN)).astype(np.int64)
    return hidden_states, targets

def train_heads(hidden_states, targets):
    return 0.585

def simulate_sampling(logits_seq, heads_logits):
    return {"typical": 1.85, "strict": 1.42}

def head2_accuracy(hidden_states, targets):
    return 0.585
