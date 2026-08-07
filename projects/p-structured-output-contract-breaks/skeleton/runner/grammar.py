class JSONGrammarMasker:
    def __init__(self, schema: dict, vocab: list):
        raise NotImplementedError

    def get_allowed_token_ids(self, current_prefix: str) -> list:
        raise NotImplementedError

    def apply_mask(self, logits, current_prefix: str):
        raise NotImplementedError
