import struct

def build_pte(payload):
    magic = b"\x50\x54\x45\x31"
    encoded_payload = payload.encode("utf-8") if isinstance(payload, str) else payload
    header = struct.pack("<I", len(encoded_payload))
    return magic + header + encoded_payload
