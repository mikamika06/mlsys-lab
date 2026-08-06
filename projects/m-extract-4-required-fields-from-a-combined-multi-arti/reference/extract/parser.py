import re

def extract_fields(log_content: str) -> dict:
    fields = {}
    m_compile = re.search(r"COMPILER_ID:\s*([^\n]+)", log_content)
    if m_compile:
        fields["compiler_id"] = m_compile.group(1).strip()
    m_graph = re.search(r"GRAPH_HASH:\s*([^\n]+)", log_content)
    if m_graph:
        fields["graph_hash"] = m_graph.group(1).strip()
    m_status = re.search(r"COMPILATION_STATUS:\s*([^\n]+)", log_content)
    if m_status:
        fields["compilation_status"] = m_status.group(1).strip()
    m_time = re.search(r"EXECUTION_TIME:\s*([^\n]+)", log_content)
    if m_time:
        fields["execution_time"] = m_time.group(1).strip()
    return fields
