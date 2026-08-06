import json

SHAPES = [
    "stream_with_usage",
    "json_schema_format",
    "tool_calls",
    "multimodal_content",
    "logprobs_topk",
    "multi_choice_n",
    "token_array_prompt",
    "stop_sequences"
]

def make_sample_request(shape_name):
    if shape_name == "stream_with_usage":
        return {
            "model": "vllm-test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
            "stream_options": {"include_usage": True}
        }
    elif shape_name == "json_schema_format":
        return {
            "model": "vllm-test-model",
            "messages": [{"role": "user", "content": "Return JSON"}],
            "response_format": {"type": "json_object"}
        }
    elif shape_name == "tool_calls":
        return {
            "model": "vllm-test-model",
            "messages": [{"role": "user", "content": "What is the weather?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
                }
            }],
            "tool_choice": "auto"
        }
    elif shape_name == "multimodal_content":
        return {
            "model": "vllm-test-model",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,1234"}}
                ]
            }]
        }
    elif shape_name == "logprobs_topk":
        return {
            "model": "vllm-test-model",
            "messages": [{"role": "user", "content": "Logprobs test"}],
            "logprobs": True,
            "top_logprobs": 5
        }
    elif shape_name == "multi_choice_n":
        return {
            "model": "vllm-test-model",
            "messages": [{"role": "user", "content": "Multiple choices"}],
            "n": 3
        }
    elif shape_name == "token_array_prompt":
        return {
            "model": "vllm-test-model",
            "prompt": [101, 2054, 2003, 102]
        }
    elif shape_name == "stop_sequences":
        return {
            "model": "vllm-test-model",
            "messages": [{"role": "user", "content": "Stop test"}],
            "stop": ["\n", "END"]
        }
    raise ValueError(f"Unknown shape: {shape_name}")

class MockServer:
    def handle(self, payload):
        shape = payload.get("_shape")
        if not shape:
            return {"error": "Missing _shape field in payload"}
        
        if shape == "stream_with_usage":
            return {
                "id": "chatcmpl-stream",
                "object": "chat.completion.chunk",
                "choices": [{"delta": {"content": "Hi"}, "index": 0}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
            }
        elif shape == "json_schema_format":
            return {
                "id": "chatcmpl-json",
                "object": "chat.completion",
                "choices": [{"message": {"role": "assistant", "content": '{"result": "ok"}'}, "index": 0}]
            }
        elif shape == "tool_calls":
            return {
                "id": "chatcmpl-tools",
                "object": "chat.completion",
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'}
                        }]
                    },
                    "index": 0
                }]
            }
        elif shape == "multimodal_content":
            return {
                "id": "chatcmpl-mm",
                "object": "chat.completion",
                "choices": [{"message": {"role": "assistant", "content": "Image seen"}, "index": 0}]
            }
        elif shape == "logprobs_topk":
            return {
                "id": "chatcmpl-logprobs",
                "object": "chat.completion",
                "choices": [{
                    "message": {"role": "assistant", "content": "A"},
                    "logprobs": {"content": [{"token": "A", "logprob": -0.1, "top_logprobs": [{"token": "A", "logprob": -0.1}]}]},
                    "index": 0
                }]
            }
        elif shape == "multi_choice_n":
            return {
                "id": "chatcmpl-n",
                "object": "chat.completion",
                "choices": [
                    {"message": {"role": "assistant", "content": "Choice 1"}, "index": 0},
                    {"message": {"role": "assistant", "content": "Choice 2"}, "index": 1},
                    {"message": {"role": "assistant", "content": "Choice 3"}, "index": 2}
                ]
            }
        elif shape == "token_array_prompt":
            return {
                "id": "cmpl-tokens",
                "object": "text_completion",
                "choices": [{"text": "Token response", "index": 0}]
            }
        elif shape == "stop_sequences":
            return {
                "id": "chatcmpl-stop",
                "object": "chat.completion",
                "choices": [{"message": {"role": "assistant", "content": "Stopped"}, "finish_reason": "stop", "index": 0}]
            }
        return {"error": "Invalid shape"}
