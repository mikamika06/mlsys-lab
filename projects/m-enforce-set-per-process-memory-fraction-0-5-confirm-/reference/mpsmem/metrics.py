from typing import List, Dict

def compute_fragmentation_ratios(history: List[Dict[str, int]]) -> List[float]:
    ratios = []
    for snapshot in history:
        driver = snapshot.get("driver_allocated", 0)
        current = snapshot.get("current_allocated", 0)
        if driver <= 0:
            ratios.append(0.0)
        else:
            diff = max(0, driver - current)
            ratios.append(float(diff) / float(driver))
    return ratios
