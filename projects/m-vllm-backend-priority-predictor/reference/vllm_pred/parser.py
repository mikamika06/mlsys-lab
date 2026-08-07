def parse_log(log_text):
    lines = log_text.split("\n")
    evaluated = []
    rejections = {}
    selected = None
    for line in lines:
        if "Evaluated backends:" in line:
            part = line.split("Evaluated backends:")[1].strip()
            evaluated = [b.strip() for b in part.strip("[]").replace("'", "").split(",")]
        elif "rejected because" in line:
            parts = line.split("Backend ")[1].split(" rejected because ")
            b_name = parts[0].strip()
            reason = parts[1].strip()
            rejections[b_name] = reason
        elif "Selected attention backend:" in line:
            selected = line.split("Selected attention backend:")[1].strip()
    return {"evaluated": evaluated, "rejections": rejections, "selected": selected}
