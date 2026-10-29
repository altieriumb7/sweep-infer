import urllib.request
import json
import time
import sys
from datasets import load_dataset

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ds = load_dataset("google/boolq", split="validation")
TARGET_INDICES = [8, 94, 67, 88, 39, 50, 43, 61, 73, 0]

def parse_answer(text: str) -> bool:
    clean = text.strip().lower()
    if clean.startswith("yes") or clean.startswith("true") or "is correct" in clean:
        return True
    if clean.startswith("no") or clean.startswith("false") or "not" in clean:
        return False
    first_yes = clean.find("yes")
    first_no = clean.find("no")
    if first_yes != -1 and (first_no == -1 or first_yes < first_no):
        return True
    if first_no != -1 and (first_yes == -1 or first_no < first_yes):
        return False
    return False

print("=" * 85)
print("BENCHMARK REALE OLLAMA (LLAMA3:LATEST 8B) SUI 10 QUESITI BOOLQ:")
print("=" * 85)

correct_ollama = 0
total_time = 0.0

for i, idx in enumerate(TARGET_INDICES):
    item = ds[idx]
    passage = item['passage']
    question = item['question']
    target = item['answer']

    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"You are a precise reading comprehension assistant. Answer the yes/no question based solely on the provided passage. "
        f"Respond with only 'Yes' or 'No', followed by a brief 1-sentence explanation.<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"Passage: {passage}\n"
        f"Question: {question}?<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    req_data = json.dumps({
        "model": "llama3:latest",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 35
        }
    }).encode('utf-8')

    t0 = time.time()
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=req_data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        res = json.loads(resp.read().decode('utf-8'))
    dt = time.time() - t0
    total_time += dt

    content = res.get('response', '').strip()
    pred = parse_answer(content)
    is_correct = (pred == target)
    if is_correct:
        correct_ollama += 1

    eval_count = res.get('eval_count', 0)
    eval_duration = res.get('eval_duration', 0) / 1e9
    tok_s = eval_count / eval_duration if eval_duration > 0 else 0
    esito = "OK" if is_correct else "ERR"

    print(f"[{i+1:<2}/10] Idx {idx:<2} | Atteso: {str(target):<5} | Ollama: {str(pred):<5} | {esito:<3} | Tempo: {dt:>5.2f}s ({tok_s:>4.2f} tok/s) | Resp: {content[:50]}")

print("-" * 85)
print(f"Accuratezza Ollama Reale: {(correct_ollama/len(TARGET_INDICES))*100:.1f}% ({correct_ollama}/{len(TARGET_INDICES)})")
print(f"Tempo Totale Ollama Reale: {total_time:.2f} s ({total_time/60:.2f} min)")
print(f"Tempo Medio per Quesito: {total_time/len(TARGET_INDICES):.2f} s")
print("=" * 85)
