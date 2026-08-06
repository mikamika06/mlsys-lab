import csv
import io


def parse_nsight_occupancy(profile_csv_lines: list) -> list:
    f = io.StringIO("".join(profile_csv_lines))
    reader = csv.DictReader(f)
    results = []
    for row in reader:
        results.append({
            "num_warps": int(row["num_warps"]),
            "achieved_occupancy": float(row["achieved_occupancy"])
        })
    return results
