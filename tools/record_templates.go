package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"text/template"
)

type ToolFunc struct {
	Name      string         `json:"name"`
	Arguments Args           `json:"arguments,omitempty"`
}

type ToolCall struct {
	ID       string   `json:"id,omitempty"`
	Type     string   `json:"type,omitempty"`
	Function ToolFunc `json:"function"`
}

type Msg struct {
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	Thinking  string     `json:"thinking,omitempty"`
	ToolName  string     `json:"tool_name,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

type Prop struct {
	Type        any      `json:"type"`
	Description string   `json:"description,omitempty"`
	Enum        []any    `json:"enum,omitempty"`
	Items       *Prop    `json:"items,omitempty"`
	Required    []string `json:"required,omitempty"`
}

type Params struct {
	Type       string          `json:"type"`
	Properties map[string]Prop `json:"properties"`
	Required   []string        `json:"required,omitempty"`
}

type Func struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  Params `json:"parameters"`
}

type Tool struct {
	Type     string `json:"type"`
	Function Func   `json:"function"`
}

// ollama gives these types a String method, so a template that prints them
// whole emits JSON rather than Go's default struct formatting.
type Tools []Tool

func (t Tools) String() string {
	b, _ := json.Marshal([]Tool(t))
	return string(b)
}

type Args map[string]any

func (a Args) String() string {
	b, _ := json.Marshal(map[string]any(a))
	return string(b)
}

type Ctx struct {
	Messages   []Msg  `json:"messages"`
	Tools      Tools  `json:"tools,omitempty"`
	System     string `json:"system,omitempty"`
	Prompt     string `json:"prompt,omitempty"`
	Response   string `json:"response,omitempty"`
	Suffix     string `json:"suffix,omitempty"`
	IsThinkSet bool   `json:"is_think_set,omitempty"`
	Think      bool   `json:"think,omitempty"`
}

type Case struct {
	Template string `json:"template"`
	Input    string `json:"input"`
	Expected string `json:"expected"`
	Err      string `json:"error,omitempty"`
}

func tsType(p Prop) string {
	if len(p.Enum) > 0 {
		parts := make([]string, 0, len(p.Enum))
		for _, e := range p.Enum {
			b, _ := json.Marshal(e)
			parts = append(parts, string(b))
		}
		return strings.Join(parts, " | ")
	}
	name := ""
	switch t := p.Type.(type) {
	case string:
		name = t
	case []any:
		parts := make([]string, 0, len(t))
		for _, x := range t {
			s, _ := x.(string)
			parts = append(parts, tsType(Prop{Type: s}))
		}
		return strings.Join(parts, " | ")
	}
	switch name {
	case "string":
		return "string"
	case "number", "integer":
		return "number"
	case "boolean":
		return "boolean"
	case "array":
		if p.Items != nil {
			return tsType(*p.Items) + "[]"
		}
		return "any[]"
	case "object":
		return "object"
	case "null":
		return "null"
	}
	return "any"
}

func funcs() template.FuncMap {
	return template.FuncMap{
		"currentDate":      func() string { return "2026-08-06" },
		"toTypeScriptType": tsType,
		"json": func(v any) string {
			b, _ := json.Marshal(v)
			return string(b)
		},
		"toJson": func(v any) string {
			b, _ := json.Marshal(v)
			return string(b)
		},
		"slice": func(v []Msg, i ...int) []Msg {
			if len(i) == 0 {
				return v
			}
			if len(i) == 1 {
				return v[i[0]:]
			}
			return v[i[0]:i[1]]
		},
	}
}

func main() {
	dir := os.Args[1]
	inputs := os.Args[2]

	raw, err := os.ReadFile(inputs)
	if err != nil {
		panic(err)
	}
	var cases map[string]Ctx
	if err := json.Unmarshal(raw, &cases); err != nil {
		panic(err)
	}

	files, _ := filepath.Glob(filepath.Join(dir, "*.template"))
	sort.Strings(files)
	out := []Case{}
	names := []string{}
	for k := range cases {
		names = append(names, k)
	}
	sort.Strings(names)

	for _, f := range files {
		body, err := os.ReadFile(f)
		if err != nil {
			continue
		}
		tmpl, err := template.New(filepath.Base(f)).Funcs(funcs()).Parse(string(body))
		if err != nil {
			out = append(out, Case{Template: filepath.Base(f), Input: "-",
				Err: "parse: " + err.Error()})
			continue
		}
		for _, name := range names {
			var buf bytes.Buffer
			c := cases[name]
			e := ""
			if err := tmpl.Execute(&buf, c); err != nil {
				e = err.Error()
			}
			out = append(out, Case{Template: filepath.Base(f), Input: name,
				Expected: buf.String(), Err: e})
		}
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	enc.SetEscapeHTML(false)
	if err := enc.Encode(map[string]any{
		"produced_by": strings.TrimSpace(os.Getenv("GOVERSION")),
		"helpers": map[string]string{
			"currentDate":      "fixed to 2026-08-06 so renderings are reproducible",
			"toTypeScriptType": "reimplemented from the documented JSON-schema mapping",
			"json/toJson":      "encoding/json Marshal",
			"slice":            "Go slice expression over messages",
		},
		"cases": out,
	}); err != nil {
		panic(err)
	}
	fmt.Fprintf(os.Stderr, "%d renderings from %d templates\n", len(out), len(files))
}
