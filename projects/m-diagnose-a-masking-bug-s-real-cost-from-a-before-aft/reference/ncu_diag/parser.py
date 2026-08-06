import csv
import io


def parse_ncu_summary(csv_text: str) -> dict:
    metrics = {}
    f = io.StringIO(csv_text.strip())
    reader = csv.DictReader(f)
    for row in reader:
        name = row.get("Metric Name", "").strip()
        val_str = row.get("Metric Value", "0").strip().replace(",", "")
        try:
            val = float(val_str)
        except ValueError:
            val = val_str
        if name:
            metrics[name] = val
    return metrics
