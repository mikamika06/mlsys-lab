import csv
import io

def extract_sol_metrics(csv_content, kernel_names, metric_name):
    f = io.StringIO(csv_content.strip())
    reader = csv.DictReader(f)
    results = {}
    for row in reader:
        k_name = row.get("Kernel Name", "").strip()
        m_name = row.get("Metric Name", "").strip()
        m_val = row.get("Metric Value", "")
        if k_name in kernel_names and m_name == metric_name:
            try:
                results[k_name] = float(m_val.replace(",", ""))
            except ValueError:
                results[k_name] = m_val
    return results
