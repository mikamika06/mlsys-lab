import csv
import io

def diff_basic_full(basic_csv, full_csv):
    def get_sections(content):
        f = io.StringIO(content.strip())
        reader = csv.DictReader(f)
        sections = set()
        for row in reader:
            sec = row.get("Section Name", "").strip()
            if sec:
                sections.add(sec)
        return sections
    basic_sections = get_sections(basic_csv)
    full_sections = get_sections(full_csv)
    return sorted(list(full_sections - basic_sections))
