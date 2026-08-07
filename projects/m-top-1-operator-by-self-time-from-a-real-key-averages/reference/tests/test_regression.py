import sys
sys.path.insert(0, ".")
from profops.ops import row_count_delta_across_batch_sizes

def test_row_count_delta_non_zero():
    tables = {
        8: [{"name": "op1", "self_cpu_time_total": 10}],
        16: [{"name": "op1", "self_cpu_time_total": 10}, {"name": "op2", "self_cpu_time_total": 20}]
    }
    delta = row_count_delta_across_batch_sizes(tables)
    assert delta != 0
