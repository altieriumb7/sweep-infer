import json

path = r"C:\Users\Umberto\.gemini\antigravity-cli\brain\3e583f1a-175c-4581-831a-4c5238e97eff\.system_generated\logs\transcript_full.jsonl"
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        if "fig3_memory_telemetry.pdf" in line and "plt." in line:
            obj = json.loads(line)
            tc = obj.get("tool_calls", [])
            for c in tc:
                content = c.get("args", {}).get("Content", "")
                if "fig3" in content:
                    print("Found plotting code in step", obj.get("step_index"))
                    with open(r"C:\Users\Umberto\synaptic-fold-v2\scripts\original_fig3_fig4_code.py", "w", encoding="utf-8") as out:
                        out.write(content)
                    break
