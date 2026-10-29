import urllib.request
import json
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

url = 'http://127.0.0.1:8089/tokenize'
payload = {'content': 'Ciao mondo'}
req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
try:
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode('utf-8'))
        print('TOKENIZE RESULT:', res)
except Exception as e:
    print('TOKENIZE ERR:', e)

url_detok = 'http://127.0.0.1:8089/detokenize'
payload_detok = {'tokens': [15339, 4429]}
req_detok = urllib.request.Request(url_detok, data=json.dumps(payload_detok).encode('utf-8'), headers={'Content-Type': 'application/json'})
try:
    with urllib.request.urlopen(req_detok) as resp:
        res = json.loads(resp.read().decode('utf-8'))
        print('DETOKENIZE RESULT:', res)
except Exception as e:
    print('DETOKENIZE ERR:', e)
