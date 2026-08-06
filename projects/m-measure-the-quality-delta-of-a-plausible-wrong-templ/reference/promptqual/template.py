import json


class PromptTemplate:

    def __init__(
        self,
        bos_token="<s>",
        eos_token="</s>",
        user_prefix="[USER]",
        user_suffix="[/USER]",
        assistant_prefix="[ASSISTANT]",
        assistant_suffix="[/ASSISTANT]",
        system_prefix="[SYSTEM]",
        system_suffix="[/SYSTEM]",
        stop_sequences=None,
    ):
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.user_prefix = user_prefix
        self.user_suffix = user_suffix
        self.assistant_prefix = assistant_prefix
        self.assistant_suffix = assistant_suffix
        self.system_prefix = system_prefix
        self.system_suffix = system_suffix
        self.stop_sequences = stop_sequences if stop_sequences is not None else [eos_token]

    def render(self, system=None, messages=None, tools=None):
        out = []
        if self.bos_token:
            out.append(self.bos_token)

        if system:
            out.append(f"{self.system_prefix}{system}{self.system_suffix}")

        if tools:
            tools_str = json.dumps(tools, sort_keys=True)
            out.append(f"[TOOLS]{tools_str}[/TOOLS]")

        if messages:
            for m in messages:
                role = m.get("role")
                content = m.get("content", "")
                if role == "user":
                    out.append(f"{self.user_prefix}{content}{self.user_suffix}")
                elif role == "assistant":
                    out.append(f"{self.assistant_prefix}{content}{self.assistant_suffix}")

        return "".join(out)


def render_and_validate_tools(template, tools, sample_messages):
    rendered = template.render(system=None, messages=sample_messages, tools=tools)

    if "[TOOLS]" not in rendered or "[/TOOLS]" not in rendered:
        return {
            "valid": False,
            "error": "Missing [TOOLS] tags in rendered prompt",
            "rendered": rendered,
            "parsed_tools": [],
        }

    try:
        start = rendered.index("[TOOLS]") + len("[TOOLS]")
        end = rendered.index("[/TOOLS]")
        raw_json = rendered[start:end]
        parsed = json.loads(raw_json)
    except Exception as e:
        return {
            "valid": False,
            "error": f"JSON parse failure: {str(e)}",
            "rendered": rendered,
            "parsed_tools": [],
        }

    if len(parsed) != len(tools):
        return {
            "valid": False,
            "error": f"Tool count mismatch: expected {len(tools)}, got {len(parsed)}",
            "rendered": rendered,
            "parsed_tools": parsed,
        }

    return {
        "valid": True,
        "error": None,
        "rendered": rendered,
        "parsed_tools": parsed,
    }


def check_stop_sequences(template):
    issues = []
    stops = template.stop_sequences or []

    if not stops:
        issues.append("No stop sequences defined")

    if template.eos_token and template.eos_token not in stops:
        issues.append(f"eos_token '{template.eos_token}' not in stop_sequences")

    if template.assistant_suffix and template.assistant_suffix not in stops:
        issues.append(f"assistant_suffix '{template.assistant_suffix}' not in stop_sequences")

    for s in stops:
        if not s or not isinstance(s, str):
            issues.append(f"Invalid stop sequence: {s}")

    return {
        "can_terminate": len(issues) == 0,
        "issues": issues,
        "effective_stops": stops,
    }
