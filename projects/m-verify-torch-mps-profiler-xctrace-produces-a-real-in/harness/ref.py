SAMPLE_XML = """<trace>
    <signpost name="DispatchKernel" start="1.0" duration="5.0" subsystem="PyTorchMPS"/>
    <signpost name="ExecuteGraph" start="7.0" duration="12.0" subsystem="PyTorchMPS"/>
    <signpost name="SyncWait" start="20.0" duration="2.0" subsystem="PyTorchMPS"/>
</trace>"""

CPU_DURATIONS = [10.0, 15.0, 5.0]
MPS_DURATIONS = [2.0, 3.0, 1.0]

MLX_TIMES = [1.5, 1.6, 1.4]
TORCH_TIMES = [4.5, 4.8, 4.2]
