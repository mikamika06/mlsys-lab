def strip_flex_ops(ops):
    return [o.replace("SELECT_TF_OPS:", "NATIVE_") if "SELECT_TF_OPS" in o else o for o in ops]
