import urllib.request
import json
import re
import sys

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3:latest"

test_p = "In generative artificial intelligence, autoregressive transformer inference exhibits extreme memory-bound characteristics during the decoding phase. While the prompt processing (prefill) phase benefits from high compute density via batched matrix-matrix multiplication (GEMM), token-by-token generation is governed by matrix-vector multiplication (GEMV). For an $N$-parameter quantized model (e.g., 4.5~GB for 8B parameters at 4.5~BPW), generating each individual token necessitates reading the entire $4.5$~GB parameter tensor from main memory into CPU registers."

sys_prompt = """You are an elite scientific copywriter and systems researcher.
Your task is to rephrase and restructure the provided LaTeX paragraph to eliminate all AI statistical watermarks, repetitive cadence, and predictable n-gram patterns.
Strict constraints:
1. Output EXCLUSIVELY in English.
2. Strictly PRESERVE all LaTeX markup ($...$, \\textbf{...}, \\texttt{...}, \\cite{...}, \\ref{...}) and numerical quantities exactly as they are.
3. Radically alter sentence structure, clause ordering, and transitions to produce natural human academic prose.
4. Output ONLY the rewritten LaTeX text. Do not include quotes, markdown fences, notes, or explanations."""

payload = {
    "model": MODEL,
    "system": sys_prompt,
    "prompt": f"Rewrite this LaTeX paragraph:\n\n{test_p}",
    "stream": False,
    "options": {"temperature": 0.7, "top_p": 0.9}
}

req = urllib.request.Request(OLLAMA_URL, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(req, timeout=120) as resp:
    res = json.loads(resp.read().decode('utf-8')).get('response', '')

print("--- ORIGINAL ---")
print(test_p)
print("\n--- REWRITTEN BY LOCAL OLLAMA ---")
print(res.strip())
