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
        raise NotImplementedError

    def render(self, system=None, messages=None, tools=None):
        raise NotImplementedError


def render_and_validate_tools(template, tools, sample_messages):
    raise NotImplementedError


def check_stop_sequences(template):
    raise NotImplementedError
