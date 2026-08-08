VALID_STREAMS = [
    b"data: {\"choices\": [{\"text\": \"hello\"}]}\n\n",
    b"event: update\ndata: {\"status\": \"running\"}\n\n",
    b"data: {\"id\": \"chatcmpl-1\"}\n\ndata: [DONE]\n\n",
    b"data: {\"a\": 1}\n\ndata: {\"a\": 2}\n\n",
    b"data: {\"choices\": []}\n\n"
]

MALFORMED_STREAMS = [
    b"data {\"choices\": []}\n\n",
    b"data: {\"choices\": []}\nmissing_colon\n\n",
    b"event: update\nno_data_field\n\n"
]

REQUESTS = [
    {"temperature": 0.5, "top_p": 0.9, "max_tokens": 100, "extra_body": {"beam_width": 4}},
    {"temperature": 1.0, "stop": ["\n"], "extra_body": {"ignore_eos": True}},
    {"max_tokens": 50}
]

CHAT_PROMPTS = [
    "<|im_start|>system\nTest<|im_end|>\nUser",
    "<|start|>user\nQuery<|end|>"
]

COMP_PROMPTS = [
    "User",
    "Query"
]
