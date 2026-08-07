import sys
sys.path.insert(0, ".")
from pte_plan.parser import parse_pte
from pte_plan.analyzer import get_peak_memory, separate_program_and_data
from pte_plan.planner import plan_buffers
import ref

def test_parser_works():
    data = ref.generate_pte_artifact()
    parsed = parse_pte(data)
    assert parsed is not None

def test_peak_memory_under_budget():
    data = ref.generate_pte_artifact()
    parsed = parse_pte(data)
    peak, _ = plan_buffers(parsed)
    budget = ref.get_device_budget(parsed)
    assert peak <= budget
