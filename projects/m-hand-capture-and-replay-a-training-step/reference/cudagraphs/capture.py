import numpy as np


class GraphCaptureSimulator:

    def __init__(self, step_trace):
        self.step_trace = step_trace
        self.static_inputs = {}
        self.static_outputs = {}
        self.recorded_graph = []
        self.memory = {}
        self.is_captured = False

    def warmup(self, inputs):
        for name, val in inputs.items():
            self.memory[name] = np.array(val, dtype=np.float64, copy=True)
        for op in self.step_trace:
            in_args = [self.memory[inp] for inp in op["inputs"]]
            if op["op"] == "add":
                res = in_args[0] + in_args[1]
            elif op["op"] == "mul":
                res = in_args[0] * in_args[1]
            elif op["op"] == "relu":
                res = np.maximum(in_args[0], 0.0)
            elif op["op"] == "copy":
                res = np.copy(in_args[0])
            self.memory[op["output"]] = res

    def capture(self, inputs):
        for name, val in inputs.items():
            self.static_inputs[name] = np.array(val, dtype=np.float64, copy=True)
            self.memory[name] = self.static_inputs[name]

        self.recorded_graph = []
        for op in self.step_trace:
            self.recorded_graph.append(
                {
                    "op": op["op"],
                    "inputs": list(op["inputs"]),
                    "output": op["output"],
                }
            )
            in_args = [self.memory[inp] for inp in op["inputs"]]
            if op["op"] == "add":
                res = in_args[0] + in_args[1]
            elif op["op"] == "mul":
                res = in_args[0] * in_args[1]
            elif op["op"] == "relu":
                res = np.maximum(in_args[0], 0.0)
            elif op["op"] == "copy":
                res = np.copy(in_args[0])
            self.memory[op["output"]] = res
            self.static_outputs[op["output"]] = self.memory[op["output"]]

        self.is_captured = True

    def replay(self, new_inputs):
        if not self.is_captured:
            raise RuntimeError("Graph not captured")

        for name, val in new_inputs.items():
            if name in self.static_inputs:
                np.copyto(self.static_inputs[name], val)

        for op in self.recorded_graph:
            in_args = [self.memory[inp] for inp in op["inputs"]]
            if op["op"] == "add":
                res = in_args[0] + in_args[1]
            elif op["op"] == "mul":
                res = in_args[0] * in_args[1]
            elif op["op"] == "relu":
                res = np.maximum(in_args[0], 0.0)
            elif op["op"] == "copy":
                res = np.copy(in_args[0])

            if op["output"] in self.memory:
                np.copyto(self.memory[op["output"]], res)
            else:
                self.memory[op["output"]] = res

        return {k: np.copy(v) for k, v in self.static_outputs.items()}
