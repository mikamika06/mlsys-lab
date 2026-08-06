import csv
import io


def parse_ncu_report(raw_data):
    reader = csv.DictReader(io.StringIO(raw_data))
    metrics = {}
    for row in reader:
        name = row.get("Metric Name") or row.get("name")
        val = row.get("Metric Value") or row.get("value")
        if name and val:
            try:
                cleaned_val = float(val.replace(",", "").strip())
            except ValueError:
                cleaned_val = val.strip()
            metrics[name.strip()] = cleaned_val
    return metrics
