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

    out = {"works_missing": 0.0, "works_present": 0.0}
    
    reader1 = ref.MockReader({"other.key": "value"})
    try:
        if mod.extract_template(reader1) is None:
            out["works_missing"] = 1.0
    except Exception as e:
        out["_note"] = f"failed on missing template: {e}"
    
    reader2 = ref.MockReader({"tokenizer.chat_template": "{{ chat }}"})
    try:
        if mod.extract_template(reader2) == "{{ chat }}":
            out["works_present"] = 1.0
    except Exception as e:
        if "_note" not in out:
            out["_note"] = f"failed on present template: {e}"
            
    return out
