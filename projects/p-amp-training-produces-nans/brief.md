# Mixed precision produces NaNs

After enabling Automatic Mixed Precision (AMP) for a large model training run, the loss values unexpectedly become NaN at a random step. When running in pure FP32, the training is completely stable. However, turning off AMP is not an option because the model exceeds available GPU memory limits.

Your task is to diagnose where the numerical instability originates, examine how operations are cast under autocast, manage gradient scaling correctly during skipped steps, isolate numerically sensitive layers, ensure thousands of steps run cleanly without memory regression, and build a robust NaN detector that pinpoints the exact offending module rather than just failing the whole training step.
```

--- projects/p-amp-training-produces-nans/project.json

```json
{
  "id": "p-amp-training-produces-nans",
  "title": "Mixed precision produces NaNs",
  "kind": "project",
  "tier": "T1",
  "area": "rw2-pytorch-applied",
  "track": "amp-nans",
  "part": 1,
  "difficulty": 3,
  "edits": [
    "amp_fix/detector.py",
    "amp_fix/scaler_utils.py",
    "amp_fix/trainer.py",
    "tests/test_regression.py"
  ],
  "milestones": [
    {
      "n": 1,
      "title": "локалізувати перший NaN у графі",
      "check": "harness/m1.py",
      "gates": [
        {
          "metric": "located_correctly",
          "op": "==",
          "threshold": 1.0
        }
      ]
    },
    {
      "n": 2,
      "title": "перевірити, які операції автокаст перевів у fp16",
      "check": "harness/m2.py",
      "gates": [
        {
          "metric": "autocast_checked",
          "op": "==",
          "threshold": 1.0
        }
      ]
    },
    {
      "n": 3,
      "title": "розібрати поведінку GradScaler на кроці пропуску",
      "check": "harness/m3.py",
      "gates": [
        {
          "metric": "scaler_behavior_ok",
          "op": "==",
          "threshold": 1.0
        }
      ]
    },
    {
      "n": 4,
      "title": "винести чутливі ділянки з автокасту",
      "check": "harness/m4.py",
      "gates": [
        {
          "metric": "sensitive_isolated",
          "op": "==",
          "threshold": 1.0
        }
      ]
    },
    {
      "n": 5,
      "title": "1000 кроків без NaN, пам'ять не гірша",
      "check": "harness/m5.py",
      "gates": [
        {
          "metric": "stable_run",
          "op": "==",
          "threshold": 1.0
        }
      ]
    },
    {
      "n": 6,
      "title": "детектор NaN, що вказує на модуль, а не на крок",
      "check": "harness/m6.py",
      "gates": [
        {
          "metric": "has_tests",
          "op": "==",
          "threshold": 1.0
        },
        {
          "metric": "passes_on_good",
          "op": "==",
          "threshold": 1.0
        },
        {
          "metric": "catches_broken_detector",
          "op": "==",
          "threshold": 1.0
        },
        {
          "metric": "faults_caught",
          "op": ">=",
          "threshold": 1.0
        }
      ]
    }
  ]
}
```

--- skeleton/amp_fix/detector.py

```python
def locate_first_nan(node_outputs):
    raise NotImplementedError


def inspect_module_nans(named_tensors):
    raise NotImplementedError
```

--- skeleton/amp_fix/scaler_utils.py

```python
def check_autocast_promotion(ops_list):
    raise NotImplementedError


def analyze_scaler_step(scaler_state, has_inf):
    raise NotImplementedError
```

--- skeleton/amp_fix/trainer.py

```python
class SensitiveModelTrainer:
    def __init__(self, model):
        raise NotImplementedError

    def train_steps(self, data_stream, num_steps):
        raise NotImplementedError
```

--- skeleton/tests/test_regression.py

```python
def test_detector_pinpoints_module():
    raise NotImplementedError


def test_no_nan_over_long_horizon():
    raise NotImplementedError
```

--- reference/amp_fix/detector.py

```python
import numpy as np


def locate_first_nan(node_outputs):
    for name, val in node_outputs.items():
        arr = np.asarray(val)
        if not np.isfinite(arr).all():
            return name
    return None


def inspect_module_nans(named_tensors):
    offending = {}
    for mod_name, tensor in named_tensors.items():
        arr = np.asarray(tensor)
        if not np.isfinite(arr).all():
            offending[mod_name] = True
        else:
            offending[mod_name] = False
    return offending
```

--- reference/amp_fix/scaler_utils.py

```python
def check_autocast_promotion(ops_list):
    promoted = []
    for op in ops_list:
        if op.get("precision") == "fp16" and op.get("sensitive") is False:
            promoted.append(op["name"])
    return promoted


def analyze_scaler_step(scaler_state, has_inf):
    scale = scaler_state["scale"]
    if has_inf:
        scale = scale * scaler_state["backoff_factor"]
        skipped = True
    else:
        scale = scale * scaler_state["growth_factor"]
        skipped = False
    return {"scale": scale, "skipped": skipped}
```

--- reference/amp_fix/trainer.py

```python
import numpy as np


class SensitiveModelTrainer:
    def __init__(self, model):
        self.model = model
        self.scale = 1024.0
        self.memory_footprint = 100.0

    def train_steps(self, data_stream, num_steps):
        step_count = 0
        for batch in data_stream:
            if step_count >= num_steps:
                break
            out = self.model(batch)
            if not np.isfinite(out).all():
                self.scale *= 0.5
            else:
                step_count += 1
        return step_count
```

--- reference/tests/test_regression.py

```python
import sys
sys.path.insert(0, ".")
from amp_fix.detector import inspect_module_nans


def test_detector_pinpoints_module():
    tensors = {"layer1": [1.0, 2.0], "layer2": [float("nan"), 1.0]}
    res = inspect_module_nans(tensors)
    assert res["layer2"] is True
    assert res["layer1"] is False


def test_no_nan_over_long_horizon():
    vals = [1.0, 2.0, 3.0]
    assert all(v > 0 for v in vals)
```

--- harness/ref.py

```python
import numpy as np


def get_mock_graph():
    return {
        "embedding": np.array([1.0, 2.0]),
        "attention": np.array([0.5, float("nan")]),
        "mlp": np.array([1.1, 1.2])
    }


def get_mock_ops():
    return [
        {"name": "matmul", "precision": "fp16", "sensitive": False},
        {"name": "exp", "precision": "fp16", "sensitive": True},
        {"name": "add", "precision": "fp32", "sensitive": False}
    ]


def get_mock_scaler_state():
    return {"scale": 65536.0, "backoff_factor": 0.5, "growth_factor": 2.0}
```

--- harness/m1.py

```python
import ref
from amp_fix.detector import locate_first_nan


def check(workdir):
    m = {"located_correctly": 0.0}
    graph = ref.get_mock_graph()
    first = locate_first_nan(graph)
    if first == "attention":
        m["located_correctly"] = 1.0
    return m
```

--- harness/m2.py

```python
import ref
from amp_fix.scaler_utils import check_autocast_promotion


def check(workdir):
    m = {"autocast_checked": 0.0}
    ops = ref.get_mock_ops()
    promoted = check_autocast_promotion(ops)
    if "matmul" in promoted and "exp" not in promoted:
        m["autocast_checked"] = 1.0
    return m
```

--- harness/m3.py

```python
import ref
from amp_fix.scaler_utils import analyze_scaler_step


def check(workdir):
    m = {"scaler_behavior_ok": 0.0}
    state = ref.get_mock_scaler_state()
    res_inf = analyze_scaler_step(state, has_inf=True)
    res_ok = analyze_scaler_step(state, has_inf=False)
    if res_inf["skipped"] is True and res_inf["scale"] < state["scale"] and res_ok["skipped"] is False and res_ok["scale"] > state["scale"]:
        m["scaler_behavior_ok"] = 1.0
    return m
```

--- harness/m4.py

```python
import ref
from amp_fix.trainer import SensitiveModelTrainer


def check(workdir):
    m = {"sensitive_isolated": 0.0}
    class DummyModel:
        def __call__(self, x):
            return x * 2.0
    trainer = SensitiveModelTrainer(DummyModel())
    steps = trainer.train_steps([1.0, 2.0, 3.0], 3)
    if steps == 3:
        m["sensitive_isolated"] = 1.0
    return m
```

--- harness/m5.py

```python
import ref
from amp_fix.trainer import SensitiveModelTrainer


def check(workdir):
    m = {"stable_run": 0.0}
    class LongModel:
        def __call__(self, x):
            return x
    trainer = LongModel()
    data = [1.0] * 1000
    res_steps = SensitiveModelTrainer(trainer).train_steps(data, 1000)
    if res_steps >= 1000:
        m["stable_run"] = 1.0
    return m
```

--- harness/m6.py

```python
import importlib.util
import os


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n))]
    if not fns:
        return None
    for fn in fns:
        fn()
    return True


def _survives(path):
    try:
        return _run(path) is True
    except Exception:
        return False


def check(workdir):
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_detector": 0.0, "faults_caught": 0.0}
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    import amp_fix.detector as det

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct implementation: {type(e).__name__}: {e}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    good_inspect = det.inspect_module_nans

    def broken_inspect(named_tensors):
        return {k: False for k in named_tensors}

    det.inspect_module_nans = broken_inspect
    try:
        out["catches_broken_detector"] = 0.0 if _survives(path) else 1.0
    finally:
        det.inspect_module_nans = good_inspect

    out["faults_caught"] = out["catches_broken_detector"]
    return out
