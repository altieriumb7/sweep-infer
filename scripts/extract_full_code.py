import json

path = r"C:\Users\Umberto\.gemini\antigravity-cli\brain\3e583f1a-175c-4581-831a-4c5238e97eff\.system_generated\logs\transcript_full.jsonl"
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        if obj.get("step_index") in [983, 1047]:
            tc = obj.get("tool_calls", [])
            for c in tc:
                content = c.get("args", {}).get("Content", "")
                if "plt." in content:
                    print("Found plotting code in step", obj.get("step_index"))
                    with open(rf"C:\Users\Umberto\synaptic-fold-v2\scripts\step_{obj.get('step_index')}_code.py", "w", encoding="utf-8") as out:
                        out.write(content)
