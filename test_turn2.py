import urllib.request
import json
import time
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

url = 'http://127.0.0.1:8089/completion'

# Turn 1 + Turn 2
system_primer = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "Sei un assistente cognitivo in lingua italiana. Quando ti viene posto un problema o un indovinello di logica, "
    "analizza la situazione passo dopo passo. Spiega in modo chiaro e dettagliato tutti i passaggi necessari per "
    "completare il compito senza commettere errori. Nella soluzione della traversata del fiume con il contadino, "
    "il lupo, la capra e il cavolo: la barca puo trasportare solo il contadino e un solo animale o oggetto per volta. "
    "Ricorda sempre che se il lupo e la capra sono lasciati soli, il lupo mangia la capra; se la capra e il cavolo sono "
    "lasciati soli, la capra mangia il cavolo. Il contadino deve fare la traversata avanti e indietro tra le due rive.<|eot_id|>"
)

user_turn1 = (
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "Mostrami la soluzione esatta per portare tutti sani e salvi sull'altra riva.<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "Ecco la soluzione passo dopo passo:\n"
    "Fase 1: Il contadino prende la barca e si reca sulla riva opposta con la capra.\n"
    "Il contadino scende dalla barca e lascia la capra sulla riva opposta.\n"
    "Il contadino torna indietro sulla riva iniziale con la barca vuota.\n"
    "Fase 2: Il contadino prende la barca e si reca sulla riva opposta con il cavolo.<|eot_id|>"
)

user_turn2 = (
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "Spiega ora la mossa cruciale: cosa deve riportare indietro il contadino per non far mangiare il cavolo dalla capra?<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

full_prompt = system_primer + user_turn1 + user_turn2

payload = {
    'prompt': full_prompt,
    'n_predict': 150,
    'temperature': 0.0,
    'stream': False
}

t0 = time.time()
req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(req) as resp:
    res = json.loads(resp.read().decode('utf-8'))
dt = time.time() - t0

tokens = res.get('tokens_predicted', 0)
speed = tokens / dt if dt > 0 else 0
print(f"Generated text:\n{res.get('content')}")
print(f"\nTokens predicted: {tokens}")
print(f"Total time: {dt:.2f} s")
print(f"Effective generation speed: {speed:.2f} tokens/s")
timings = res.get('timings', {})
print(f"Timings: {json.dumps(timings, indent=2)}")
