from gguf_engine.importer import GGUFImporter
from gguf_engine.tokenizer import GGUFTokenizer
from gguf_engine.engine import GGUFEngine

def get_oracle_metadata():
    return {
        "version": 3,
        "architecture": "llama",
        "chat_template": "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\n' + message['content'] + '\n' }}{% endif %}{% endfor %}",
        "stop_tokens": [2, 32000]
    }
