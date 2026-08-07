import sys
sys.path.insert(0, ".")
from gguf_engine.importer import GGUFImporter
from gguf_engine.tokenizer import GGUFTokenizer
from gguf_engine.engine import GGUFEngine

def test_importer_metadata():
    imp = GGUFImporter("model.gguf")
    meta = imp.verify_metadata()
    assert "architecture" in meta

def test_chat_template():
    imp = GGUFImporter("model.gguf")
    tok = GGUFTokenizer(imp.verify_metadata())
    res = tok.apply_chat_template([{"role": "user", "content": "hello"}])
    assert "<|user|>" in res

def test_engine_generation():
    imp = GGUFImporter("model.gguf")
    tok = GGUFTokenizer(imp.verify_metadata())
    engine = GGUFEngine(imp, tok)
    out = engine.generate([{"role": "user", "content": "test"}])
    assert len(out) > 0
