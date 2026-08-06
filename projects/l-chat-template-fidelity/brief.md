# The prompt the model actually sees

An evaluation harness scores a model through an OpenAI-shaped API. A serving
team scores the same model through their own runner. The numbers disagree by
four points and nobody can say why, because the two paths do not build the same
prompt: one of them drops a space before `[INST]`, or emits the tool block in a
different place, or renders an empty assistant turn the other one omits.

The model never sees your messages. It sees a string. That string is produced by
a Go `text/template` living inside the runner, and until you can reproduce it
byte for byte you are guessing about what you evaluated.

You are building `gotmpl`: enough of Go's `text/template` to render real chat
templates exactly. Pure Python, standard library only.

## The fixtures

`projects/_fixtures/ollama/` holds four chat templates copied out of ollama's
local blob store — two from a Mistral-family model, two from gpt-oss. They are
not simplified. The gpt-oss pair is 7 kB of harmony format that builds a
TypeScript type for every tool the caller passes.

`projects/_fixtures/templates/renderings.json` is what Go produced from those
templates for twenty different conversations: single turns, tool calls and their
results, empty content, unicode, a trailing assistant turn. It was recorded by
running Go 1.25's own `text/template`, so it is not my opinion about what the
template means.

`semantics/` holds fourteen small templates that isolate one rule each — the
trim markers, `range` with an index variable, assignment to an outer variable,
map iteration order, pipelines into a function — rendered against the same
twenty inputs. That is 280 more recorded outputs, and they are the ones that
tell you *which* rule you got wrong when a big template comes out crooked.

Together: 360 recorded renderings. A milestone passes when every one of the
renderings it selects matches byte for byte. There is no partial credit, because
a template that is 99% right produces a prompt that is wrong.

## The rules that actually bite

**Whitespace.** `{{-` eats the whitespace before the action and `-}}` eats what
follows. Templates are written to be readable, so nearly every line uses them,
and getting this wrong shifts the whole output by a newline.

**Map order.** Ranging over a map visits keys in sorted order. Go guarantees
this for templates specifically, and the recorded tool schemas depend on it.

**Scope.** Inside a `range`, `.` is the element; `$` is still the top-level
data. A template that needs both — "is this the last user message *and* were
tools supplied" — reads `$.Tools` from inside the loop.

**Assignment versus definition.** `$x := v` makes a new variable in this scope;
`$x = v` writes to the one already there. The first real template opens by
scanning the messages to find the last user turn, and it only works because the
assignment reaches out of the loop.

**Truth.** Zero, empty string, empty slice, empty map and nil are all false.
`{{ if $i }}` is how a template writes "not the first iteration".

**Printing.** Not everything prints as its container. Two values in these
fixtures come from Go types that carry their own `String` method and emit
compact JSON. A key the data does not have stands for a zero-valued field and
prints as nothing at all.

## Functions

The runner supplies functions the template can call, and so does the harness:
`currentDate`, `toTypeScriptType`, `json`, `toJson`, `slice`. They arrive in the
`funcs` argument of `render`; you implement the built-ins that Go itself
provides — `eq`, `ne`, `lt`, `le`, `gt`, `ge`, `not`, `len`, `index`, `and`,
`or`. `eq` takes more than two arguments. `and` and `or` return the operand that
decided the result, not a boolean.

## Milestones

1. The tokeniser and the simple actions, including both trim markers.
2. Pipelines, scoping, map order, assignment.
3. A real chat template on plain conversations.
4. The same template once tools and tool results appear.
5. Harmony: nested ranges, tool schemas rendered as TypeScript.
6. Malformed templates raise `TemplateError`, and your own regression suite is
   able to fail — the harness hands it a renderer that strips the output, and a
   suite that still reports success is not a suite.
