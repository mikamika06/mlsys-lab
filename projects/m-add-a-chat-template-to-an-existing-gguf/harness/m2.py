import sys
import ref

def check(workdir):
    sys.path.insert(0, workdir)
    try:
        import gguf_chat.modifier as mod
    except ImportError:
        return {"_note": "could not import gguf_chat.modifier"}
    finally:
        sys.path.pop(0)

    out = {"sets_new": 0.0, "backs_up_current": 0.0, "preserves_old_backup": 0.0}

    try:
        reader1 = ref.MockReader({})
        writer1 = ref.MockWriter()
        mod.set_chat_template(reader1, writer1, "new")
        if writer1.kv.get("tokenizer.chat_template") == "new" and "tokenizer.chat_template.backup" not in writer1.kv:
            out["sets_new"] = 1.0

        reader2 = ref.MockReader({"tokenizer.chat_template": "old"})
        writer2 = ref.MockWriter()
        mod.set_chat_template(reader2, writer2, "new")
        if writer2.kv.get("tokenizer.chat_template") == "new" and writer2.kv.get("tokenizer.chat_template.backup") == "old":
            out["backs_up_current"] = 1.0

        reader3 = ref.MockReader({
            "tokenizer.chat_template": "old", 
            "tokenizer.chat_template.backup": "oldest"
        })
        writer3 = ref.MockWriter()
        mod.set_chat_template(reader3, writer3, "new")
        if writer3.kv.get("tokenizer.chat_template") == "new" and writer3.kv.get("tokenizer.chat_template.backup") == "oldest":
            out["preserves_old_backup"] = 1.0
    except Exception as e:
        out["_note"] = f"runtime error: {e}"

    return out
