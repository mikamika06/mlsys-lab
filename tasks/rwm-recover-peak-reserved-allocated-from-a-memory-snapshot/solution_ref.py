def recover_memory_peaks(snapshot):
    reserved = sum(segment["size"] for segment in snapshot["segments"])
    current_allocated = 0
    peak_reserved = 0
    peak_allocated = 0
    largest_allocation = 0

    for segment in snapshot["segments"]:
        for block in segment["blocks"]:
            for event in block["events"]:
                if event == "alloc":
                    current_allocated += block["size"]
                    largest_allocation = max(
                        largest_allocation, block["size"]
                    )
                elif event == "free":
                    current_allocated -= block["size"]

                peak_reserved = max(peak_reserved, reserved)
                peak_allocated = max(peak_allocated, current_allocated)

    return peak_reserved, peak_allocated, largest_allocation
