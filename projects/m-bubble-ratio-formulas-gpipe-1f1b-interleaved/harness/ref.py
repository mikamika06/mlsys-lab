import pipelib.formulas as formulas
import pipelib.memory as memory
import pipelib.schedule as schedule

CONFIGS = [
    {"p": 2, "m": 8, "v": 1},
    {"p": 4, "m": 16, "v": 1},
    {"p": 4, "m": 16, "v": 2},
    {"p": 8, "m": 32, "v": 1},
    {"p": 8, "m": 32, "v": 4},
    {"p": 16, "m": 64, "v": 2},
]


def eval_formulas(cfg):
    p, m, v = cfg["p"], cfg["m"], cfg["v"]
    return {
        "gpipe": formulas.gpipe_bubble_ratio(p, m),
        "f1b": formulas.f1b_bubble_ratio(p, m),
        "interleaved": formulas.interleaved_1f1b_bubble_ratio(p, m, v),
    }


def eval_schedules(cfg):
    p, m = cfg["p"], cfg["m"]
    gpipe = schedule.ScheduleGPipe(p, m, f_cost=1.0, b_cost=2.0).run()
    f1b = schedule.Schedule1F1B(p, m, f_cost=1.0, b_cost=2.0).run()
    return {
        "gpipe_makespan": gpipe["makespan"],
        "gpipe_bubble": gpipe["bubble_ratio"],
        "f1b_makespan": f1b["makespan"],
        "f1b_bubble": f1b["bubble_ratio"],
    }


def eval_memory(cfg):
    p, m, v = cfg["p"], cfg["m"], cfg["v"]
    peaks_1f1b = memory.measure_peak_inflight_microbatches(p, m, v=1, schedule_type="1f1b")
    mem_1f1b = memory.estimate_activation_memory_mb(peaks_1f1b, bytes_per_microbatch=1048576.0 * 16)
    return {
        "peaks_1f1b": peaks_1f1b,
        "mem_1f1b": mem_1f1b,
    }
