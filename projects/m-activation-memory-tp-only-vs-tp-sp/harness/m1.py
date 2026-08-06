import sys
import ref

def rel_err(expected, actual):
    if expected == 0:
        return 0.0 if actual == 0 else 1.0
    return abs(expected - actual) / float(expected)

def check(workdir):
    sys.path.insert(0, workdir)
    try:
        import tp_sp.memory as mem
    except Exception as e:
        return {"mem_err": 1.0, "comm_err": 1.0, "_note": str(e)}
    
    mem_errs, comm_errs = [], []
    for cfg in ref.CONFIGS:
        try:
            m_tp_got = mem.activation_memory_per_layer(*cfg, False)
            m_sp_got = mem.activation_memory_per_layer(*cfg, True)
            c_tp_got = mem.forward_communication_volume(*cfg, False)
            c_sp_got = mem.forward_communication_volume(*cfg, True)
        except Exception as e:
            return {"mem_err": 1.0, "comm_err": 1.0, "_note": f"crash: {e}"}
        
        m_tp_ref = ref.activation_memory_per_layer(*cfg, False)
        m_sp_ref = ref.activation_memory_per_layer(*cfg, True)
        c_tp_ref = ref.forward_communication_volume(*cfg, False)
        c_sp_ref = ref.forward_communication_volume(*cfg, True)

        mem_errs.append(rel_err(m_tp_ref, m_tp_got))
        mem_errs.append(rel_err(m_sp_ref, m_sp_got))
        comm_errs.append(rel_err(c_tp_ref, c_tp_got))
        comm_errs.append(rel_err(c_sp_ref, c_sp_got))

    return {
        "mem_err": sum(mem_errs) / len(mem_errs),
        "comm_err": sum(comm_errs) / len(comm_errs)
    }
