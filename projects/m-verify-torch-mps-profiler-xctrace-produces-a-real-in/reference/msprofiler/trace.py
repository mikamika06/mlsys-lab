import xml.etree.ElementTree as ET


def parse_signposts(xml_data: str) -> list:
    root = ET.fromstring(xml_data)
    signposts = []
    for elem in root.findall(".//signpost"):
        name = elem.get("name")
        start = float(elem.get("start", 0))
        duration = float(elem.get("duration", 0))
        subsystem = elem.get("subsystem", "")
        signposts.append({
            "name": name,
            "start": start,
            "duration": duration,
            "subsystem": subsystem
        })
    return signposts
