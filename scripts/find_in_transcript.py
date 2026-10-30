import json

path = r"C:\Users\Umberto\.gemini\antigravity-cli\brain\3e583f1a-175c-4581-831a-4c5238e97eff\.system_generated\logs\transcript.jsonl"
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        if "fig3_memory_telemetry" in line or "fig4_cross_platform" in line:
            obj = json.loads(line)
            print("Step:", obj.get("step_index"), obj.get("type"))
