class ScheduleGPipe:
    """GPipe schedule simulator."""

    def __init__(self, p: int, m: int, f_cost: float = 1.0, b_cost: float = 2.0):
        self.p = p
        self.m = m
        self.f_cost = f_cost
        self.b_cost = b_cost

    def run(self) -> dict:
        rank_events = [[] for _ in range(self.p)]
        f_end = [[0.0] * self.m for _ in range(self.p)]
        b_end = [[0.0] * self.m for _ in range(self.p)]

        for microbatch in range(self.m):
            for stage in range(self.p):
                prev_stage_time = f_end[stage - 1][microbatch] if stage > 0 else 0.0
                prev_microbatch_time = f_end[stage][microbatch - 1] if microbatch > 0 else 0.0
                start_time = max(prev_stage_time, prev_microbatch_time)
                end_time = start_time + self.f_cost
                f_end[stage][microbatch] = end_time
                rank_events[stage].append({
                    "type": "forward",
                    "microbatch": microbatch,
                    "start": start_time,
                    "end": end_time
                })

        for microbatch in range(self.m - 1, -1, -1):
            for stage in range(self.p - 1, -1, -1):
                prev_stage_time = b_end[stage + 1][microbatch] if stage < self.p - 1 else f_end[self.p - 1][self.m - 1]
                prev_microbatch_time = b_end[stage][microbatch + 1] if microbatch < self.m - 1 else 0.0
                start_time = max(prev_stage_time, prev_microbatch_time)
                end_time = start_time + self.b_cost
                b_end[stage][microbatch] = end_time
                rank_events[stage].append({
                    "type": "backward",
                    "microbatch": microbatch,
                    "start": start_time,
                    "end": end_time
                })

        makespan = max(b_end[stage][0] for stage in range(self.p))
        total_work = self.m * (self.f_cost + self.b_cost)
        bubble_time = (makespan - total_work) * self.p
        bubble_ratio = bubble_time / (makespan * self.p) if makespan > 0 else 0.0

        return {
            "makespan": makespan,
            "bubble_ratio": bubble_ratio,
            "events": rank_events
        }


class Schedule1F1B:
    """1F1B schedule simulator."""

    def __init__(self, p: int, m: int, f_cost: float = 1.0, b_cost: float = 2.0):
        self.p = p
        self.m = m
        self.f_cost = f_cost
        self.b_cost = b_cost

    def run(self) -> dict:
        rank_events = [[] for _ in range(self.p)]
        f_end = [[0.0] * self.m for _ in range(self.p)]
        b_end = [[0.0] * self.m for _ in range(self.p)]

        rank_time = [0.0] * self.p
        f_completed = [0] * self.p
        b_completed = [0] * self.p

        work_remaining = True
        while work_remaining:
            work_remaining = False
            for stage in range(self.p):
                can_do_b = False
                mb_b = b_completed[stage]
                if mb_b < self.m and f_completed[stage] > mb_b:
                    if stage == self.p - 1 or b_completed[stage + 1] > mb_b:
                        can_do_b = True

                can_do_f = False
                mb_f = f_completed[stage]
                if mb_f < self.m:
                    if stage == 0 or f_completed[stage - 1] > mb_f:
                        can_do_f = True

                warmup_target = min(self.p - stage, self.m)
                prefer_f = f_completed[stage] < warmup_target

                action = None
                if prefer_f and can_do_f:
                    action = "F"
                elif can_do_b:
                    action = "B"
                elif can_do_f:
                    action = "F"

                if action == "F":
                    work_remaining = True
                    mb = f_completed[stage]
                    dep_time = f_end[stage - 1][mb] if stage > 0 else 0.0
                    start_time = max(rank_time[stage], dep_time)
                    end_time = start_time + self.f_cost
                    f_end[stage][mb] = end_time
                    rank_time[stage] = end_time
                    f_completed[stage] += 1
                    rank_events[stage].append({
                        "type": "forward",
                        "microbatch": mb,
                        "start": start_time,
                        "end": end_time
                    })
                elif action == "B":
                    work_remaining = True
                    mb = b_completed[stage]
                    dep_time = b_end[stage + 1][mb] if stage < self.p - 1 else f_end[stage][mb]
                    start_time = max(rank_time[stage], dep_time)
                    end_time = start_time + self.b_cost
                    b_end[stage][mb] = end_time
                    rank_time[stage] = end_time
                    b_completed[stage] += 1
                    rank_events[stage].append({
                        "type": "backward",
                        "microbatch": mb,
                        "start": start_time,
                        "end": end_time
                    })

        makespan = max(rank_time)
        total_work = self.m * (self.f_cost + self.b_cost)
        bubble_time = (makespan - total_work) * self.p
        bubble_ratio = bubble_time / (makespan * self.p) if makespan > 0 else 0.0

        return {
            "makespan": makespan,
            "bubble_ratio": bubble_ratio,
            "events": rank_events
        }
