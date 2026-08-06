from tp_sp.memory import activation_memory_per_layer, forward_communication_volume

def test_sp_memory_invariant():
    mem_tp = activation_memory_per_layer(1024, 4, 1024, 4, False)
    mem_sp = activation_memory_per_layer(1024, 4, 1024, 4, True)
    assert mem_sp < mem_tp
    assert mem_sp * 4 == mem_tp

def test_comm_volume_invariant():
    comm_tp = forward_communication_volume(1024, 4, 1024, 4, False)
    comm_sp = forward_communication_volume(1024, 4, 1024, 4, True)
    assert comm_tp == comm_sp
