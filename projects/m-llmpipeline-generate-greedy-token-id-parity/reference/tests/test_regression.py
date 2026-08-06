import sys

sys.path.insert(0, ".")
from inference import run_hand_rolled, run_pipeline
import ref

def test_parity_and_axis():
    model = ref.MockModel()
    prompt = [42, 100, 256]
    max_toks = 5
    hand = run_hand_rolled(model, prompt, max_toks)
    pipe = ref.MockPipeline(model).generate(prompt, max_toks)
    assert hand == pipe
