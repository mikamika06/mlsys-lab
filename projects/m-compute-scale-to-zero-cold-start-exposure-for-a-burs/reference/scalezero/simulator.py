def simulate_scale_to_zero(traffic, idle_timeout, cold_start_latency):
    state = "WARM"
    idle_counter = 0
    warmup_counter = 0

    exposed = 0
    cold_time = 0

    for req in traffic:
        if state == "WARM":
            if req == 0:
                idle_counter = 1
                if idle_counter >= idle_timeout:
                    state = "COLD"
                else:
                    state = "IDLE"
            else:
                pass

        elif state == "IDLE":
            if req > 0:
                state = "WARM"
                idle_counter = 0
            else:
                idle_counter += 1
                if idle_counter >= idle_timeout:
                    state = "COLD"

        elif state == "COLD":
            if req > 0:
                exposed += req
                warmup_counter = 1
                if warmup_counter >= cold_start_latency:
                    state = "WARM"
                else:
                    state = "WARMING"
            else:
                cold_time += 1

        elif state == "WARMING":
            exposed += req
            warmup_counter += 1
            if warmup_counter >= cold_start_latency:
                state = "WARM"

    return exposed, cold_time
