class V2ToV3Adapter:
    def __init__(self, v2_plugin):
        self.v2_plugin = v2_plugin

    def get_field_values(self):
        return getattr(self.v2_plugin, "fields", {})

    def supports_format_combination(self, pos, in_out, direction):
        return True
