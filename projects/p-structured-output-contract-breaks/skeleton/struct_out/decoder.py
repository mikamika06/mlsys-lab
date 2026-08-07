class SchemaDecoder:
    def __init__(self, schema):
        raise NotImplementedError()

    def decode(self, raw):
        raise NotImplementedError()

    def validate(self, data):
        raise NotImplementedError()
