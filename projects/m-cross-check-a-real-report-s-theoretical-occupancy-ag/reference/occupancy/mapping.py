"""Field mapping reference."""

import ref

def map_field(field_name):
    return ref.FIELDS_TO_SECTIONS.get(field_name, "Unknown")
