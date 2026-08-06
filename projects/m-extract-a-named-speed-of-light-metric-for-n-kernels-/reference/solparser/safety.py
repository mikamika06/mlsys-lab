import csv
import io

def check_replay_trustworthiness(csv_content):
    f = io.StringIO(csv_content.strip())
    reader = csv.DictReader(f)
    untrusted = []
    for row in reader:
        warn = row.get("Warning", "").lower()
        k_name = row.get("Kernel Name", "").strip()
        if "replay" in warn or "mismatch" in warn:
            if k_name not in untrusted:
                untrusted.append(k_name)
    return untrusted
