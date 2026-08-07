SAMPLE_RESPONSES = [
    {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello world",
                    "reasoning_content": "Internal thought process here"
                },
                "finish_reason": "stop"
            }
        ],
        "reasoning_content": "Top level leak"
    },
    {
        "id": "chatcmpl-2",
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": "part",
                    "reasoning_content": "thinking..."
                },
                "finish_reason": None
            }
        ]
    }
]
