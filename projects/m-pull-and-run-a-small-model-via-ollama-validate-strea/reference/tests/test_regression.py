import sys
sys.path.insert(0, ".")
from ollamautil.stream import parse_stream

def test_stream_parsing_validates_json():
    schema = {
        "type": "object",
        "properties": {"status": {"type": "string"}, "value": {"type": "integer"}},
        "required": ["status"]
    }
    chunks = ['{"status": "ok"}', 'INVALID_JSON', '{"other": 123}']
    res = parse_stream(chunks, schema)
    assert len(res) == 1
    assert res[0] == '{"status": "ok"}'
