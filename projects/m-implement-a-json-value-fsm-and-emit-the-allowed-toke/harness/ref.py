VOCAB = ["{", "}", "[", "]", "\"", '"key"', ":", "1", "2", "true", "false", "null", ",", "foo"]
SCHEMAS = [
    {"type": "object", "properties": {"a": {"type": "integer"}} },
    {"type": "object", "$ref": "#/definitions/Bar"}
]
