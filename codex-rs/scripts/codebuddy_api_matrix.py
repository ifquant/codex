import json, os, pathlib, urllib.request, urllib.error, uuid

out = pathlib.Path(os.environ.get("CODEBUDDY_MATRIX_OUT", "/tmp/codebuddy-matrix-live"))
out.mkdir(parents=True, exist_ok=True)
url = "https://copilot.tencent.com/v2/chat/completions"
key = os.environ.get("CODEBUDDY_API_KEY")
if not key:
    raise SystemExit("CODEBUDDY_API_KEY is required")
base = {
    "model": "deepseek-v4.1-flash",
    "stream": True,
    "stream_options": {"include_usage": True},
    "reasoning_effort": "high",
    "messages": [
        {
            "role": "system",
            "content": "You are a coding assistant. Follow the user request.",
        }
    ],
}
fn = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Return the current time.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
}
cases = {
    "text": {
        **base,
        "messages": base["messages"]
        + [{"role": "user", "content": "Reply exactly MATRIX_TEXT_OK."}],
    },
    "reasoning": {
        **base,
        "messages": base["messages"]
        + [
            {
                "role": "user",
                "content": "Briefly reason about 2+2, then answer exactly MATRIX_REASON_OK.",
            }
        ],
    },
    "function_decl": {
        **base,
        "tools": [fn],
        "tool_choice": "none",
        "messages": base["messages"]
        + [{"role": "user", "content": "Reply exactly MATRIX_FUNCTION_DECL_OK."}],
    },
    "function_call": {
        **base,
        "tools": [fn],
        "tool_choice": "auto",
        "messages": base["messages"]
        + [
            {
                "role": "user",
                "content": "You must call the get_time function now. Do not answer with text.",
            }
        ],
    },
    "custom_decl": {
        **base,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "apply_patch",
                    "description": "Apply a patch.",
                    "parameters": {
                        "type": "object",
                        "properties": {"input": {"type": "string"}},
                        "required": ["input"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": "none",
        "messages": base["messages"]
        + [{"role": "user", "content": "Reply exactly MATRIX_CUSTOM_DECL_OK."}],
    },
    "tool_followup": {
        **base,
        "messages": base["messages"]
        + [
            {"role": "user", "content": "Call get_time."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_time", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "2026-09-18 09:00"},
        ],
    },
}


def run(name, body):
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "X-Client-Request-Id": str(uuid.uuid4()),
        },
    )
    status = None
    chunks = []
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            status = r.status
            raw = r.read().decode()
        for line in raw.splitlines():
            if line.startswith("data: "):
                v = line[6:]
                if v != "[DONE]":
                    try:
                        chunks.append(json.loads(v))
                    except:
                        pass
        choices = [c for x in chunks for c in x.get("choices", [])]
        deltas = [c.get("delta", {}) for c in choices]
        result = {
            "status": status,
            "chunks": len(chunks),
            "done": raw.rstrip().endswith("data: [DONE]"),
            "finish_reasons": [
                c.get("finish_reason") for c in choices if c.get("finish_reason")
            ],
            "has_reasoning": any(d.get("reasoning_content") for d in deltas),
            "has_tool_calls": any(d.get("tool_calls") for d in deltas),
            "text": " ".join(d.get("content", "") for d in deltas if d.get("content"))[
                :300
            ],
        }
    except urllib.error.HTTPError as e:
        b = e.read().decode()
        try:
            err = json.loads(b)
        except:
            err = {"raw": b[:500]}
        result = {"status": e.code, "error": err}
    except Exception as e:
        result = {"error": type(e).__name__ + ": " + str(e)}
    (out / (name + ".json")).write_text(
        json.dumps(result, indent=2, ensure_ascii=False)
    )
    print(name, json.dumps(result, ensure_ascii=False), flush=True)


for n, b in cases.items():
    run(n, b)
(out / "summary.json").write_text(
    json.dumps(
        {n: json.loads((out / (n + ".json")).read_text()) for n in cases},
        indent=2,
        ensure_ascii=False,
    )
)
