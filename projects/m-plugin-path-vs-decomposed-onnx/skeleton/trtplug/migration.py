class V2ToV3Adapter:
    def __init__(self, v2_plugin):
        raise NotImplementedError

    def get_field_values(self):
        raise NotImplementedError

    def supports_format_combination(self, pos, in_out, direction):
        raise NotImplementedError
