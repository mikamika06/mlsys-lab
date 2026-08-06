class JSONFSM:
    def __init__(self, schema):
        raise NotImplementedError

    def step(self, token_id):
        raise NotImplementedError

    def allowed_tokens(self, vocab_tokens):
        raise NotImplementedError
