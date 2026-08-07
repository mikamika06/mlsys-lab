def scan_protobuf_fields(raw_bytes):
    pos = 0
    length = len(raw_bytes)
    fields = []

    while pos < length:
        start_pos = pos
        varint_val = 0
        shift = 0
        while pos < length:
            b = raw_bytes[pos]
            pos += 1
            varint_val |= (b & 0x7F) << shift
            if (b & 0x80) == 0:
                break
            shift += 7

        field_number = varint_val >> 3
        wire_type = varint_val & 0x07

        if wire_type == 0:
            val = 0
            s = 0
            while pos < length:
                b = raw_bytes[pos]
                pos += 1
                val |= (b & 0x7F) << s
                if (b & 0x80) == 0:
                    break
                s += 7
            fields.append({"field_number": field_number, "wire_type": wire_type, "length": pos - start_pos})
        elif wire_type == 1:
            pos += 8
            fields.append({"field_number": field_number, "wire_type": wire_type, "length": pos - start_pos})
        elif wire_type == 2:
            l_val = 0
            s = 0
            while pos < length:
                b = raw_bytes[pos]
                pos += 1
                l_val |= (b & 0x7F) << s
                if (b & 0x80) == 0:
                    break
                s += 7
            pos += l_val
            fields.append({"field_number": field_number, "wire_type": wire_type, "length": pos - start_pos})
        elif wire_type == 5:
            pos += 4
            fields.append({"field_number": field_number, "wire_type": wire_type, "length": pos - start_pos})
        else:
            break

    return fields
