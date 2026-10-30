import json

path = r"C:\Users\Umberto\.gemini\antigravity-cli\brain\3e583f1a-175c-4581-831a-4c5238e97eff\.system_generated\logs\transcript.jsonl"
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        if obj.get("step_index") in [1017, 1071]:
            tc = obj.get("tool_calls", [])
            for c in tc:
                args = c.get("args", {})
                for k, v in args.items():
                    if isinstance(v, str) and ("fig3" in v or "fig4" in v):
                        print(f"Step {obj.get('step_index')} tool call {c.get('name')}: {v[:300]}")
                        with open(f"step_{obj.get('step_index')}.txt", "w", encoding="utf-8") as out:
                            out.write(v)
