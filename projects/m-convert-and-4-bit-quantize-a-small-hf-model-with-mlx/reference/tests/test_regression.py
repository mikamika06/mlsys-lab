from mlxflow.server_check import validate_openai_schema

def test_valid_payload():
    payload = {
        "object": "chat.completion",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello world"
                }
            }
        ]
    }
    assert validate_openai_schema(payload) is True

def test_invalid_payload_missing_choices():
    payload = {
        "object": "chat.completion",
        "choices": []
    }
    assert validate_openai_schema(payload) is False
