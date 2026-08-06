import xml.etree.ElementTree as ET


def parse_ir_xml(xml_content):
    root = ET.fromstring(xml_content)
    ops = []
    for layer in root.iter("layer"):
        op_type = layer.get("type")
        name = layer.get("name")
        if op_type:
            ops.append({"name": name, "type": op_type})
    return ops
