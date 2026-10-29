"""
SWEEP Ollama & OpenAI Drop-In API Server
Author: Umberto Altieri
License: MIT
"""

import argparse
import json
import time
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import datetime

from .engine import SweepEngine

GLOBAL_ENGINE = None
GLOBAL_MODEL_NAME = "llama3:latest"

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

class SweepAPIHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, data: dict):
        body = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.end_headers()

    def do_GET(self):
        if self.path in ('/api/version', '/version'):
            self._send_json(200, {"version": "0.4.0-sweep"})
        elif self.path in ('/api/tags', '/v1/models'):
            self._send_json(200, {
                "models": [
                    {
                        "name": GLOBAL_MODEL_NAME,
                        "model": GLOBAL_MODEL_NAME,
                        "modified_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "size": 4920739840,
                        "details": {
                            "parent_model": "",
                            "format": "gguf",
                            "family": "llama",
                            "parameter_size": "8.0B",
                            "quantization_level": "Q4_K_M"
                        }
                    }
                ]
            })
        else:
            self._send_json(200, {"status": "SWEEP Inference Engine Online"})

    def do_POST(self):
        content_len = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_len).decode('utf-8')
        try:
            req = json.loads(post_data) if post_data else {}
        except Exception:
            self._send_json(400, {"error": "Invalid JSON"})
            return

        # 1. Ollama /api/generate
        if self.path == '/api/generate':
            prompt = req.get('prompt', '')
            stream = req.get('stream', False)
            max_tokens = req.get('options', {}).get('num_predict', 128)
            self._handle_generate(prompt, stream, max_tokens)

        # 2. Ollama /api/chat
        elif self.path == '/api/chat':
            messages = req.get('messages', [])
            stream = req.get('stream', False)
            max_tokens = req.get('options', {}).get('num_predict', 128)
            prompt = self._format_chat_messages(messages)
            self._handle_chat(prompt, stream, max_tokens)

        # 3. OpenAI /v1/chat/completions
        elif self.path == '/v1/chat/completions':
            messages = req.get('messages', [])
            stream = req.get('stream', False)
            max_tokens = req.get('max_tokens', 128)
            prompt = self._format_chat_messages(messages)
            self._handle_openai(prompt, stream, max_tokens)
        else:
            self._send_json(404, {"error": f"Endpoint {self.path} not found"})

    def _format_chat_messages(self, messages: list) -> str:
        formatted = "<|begin_of_text|>"
        for m in messages:
            role = m.get('role', 'user')
            content = m.get('content', '')
            formatted += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return formatted

    def _handle_generate(self, prompt: str, stream: bool, max_tokens: int):
        t0 = time.time()
        if not stream:
            tokens_text = []
            for piece in GLOBAL_ENGINE.generate(prompt, max_tokens=max_tokens):
                tokens_text.append(piece)
            dt = time.time() - t0
            full_text = "".join(tokens_text)
            self._send_json(200, {
                "model": GLOBAL_MODEL_NAME,
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "response": full_text,
                "done": True,
                "total_duration": int(dt * 1e9),
                "eval_count": len(tokens_text),
                "eval_duration": int(dt * 1e9)
            })
        else:
            self.send_response(200)
            self.send_header('Content-Type', 'application/x-ndjson')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            count = 0
            for piece in GLOBAL_ENGINE.generate(prompt, max_tokens=max_tokens):
                count += 1
                line = json.dumps({"model": GLOBAL_MODEL_NAME, "response": piece, "done": False}) + "\n"
                self.wfile.write(line.encode('utf-8'))
                self.wfile.flush()
            dt = time.time() - t0
            line = json.dumps({
                "model": GLOBAL_MODEL_NAME,
                "response": "",
                "done": True,
                "eval_count": count,
                "eval_duration": int(dt * 1e9)
            }) + "\n"
            self.wfile.write(line.encode('utf-8'))

    def _handle_chat(self, prompt: str, stream: bool, max_tokens: int):
        t0 = time.time()
        if not stream:
            tokens_text = []
            for piece in GLOBAL_ENGINE.generate(prompt, max_tokens=max_tokens):
                tokens_text.append(piece)
            dt = time.time() - t0
            full_text = "".join(tokens_text)
            self._send_json(200, {
                "model": GLOBAL_MODEL_NAME,
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "message": {"role": "assistant", "content": full_text},
                "done": True,
                "total_duration": int(dt * 1e9),
                "eval_count": len(tokens_text),
                "eval_duration": int(dt * 1e9)
            })
        else:
            self.send_response(200)
            self.send_header('Content-Type', 'application/x-ndjson')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            count = 0
            for piece in GLOBAL_ENGINE.generate(prompt, max_tokens=max_tokens):
                count += 1
                line = json.dumps({"model": GLOBAL_MODEL_NAME, "message": {"role": "assistant", "content": piece}, "done": False}) + "\n"
                self.wfile.write(line.encode('utf-8'))
                self.wfile.flush()
            line = json.dumps({"model": GLOBAL_MODEL_NAME, "done": True, "eval_count": count}) + "\n"
            self.wfile.write(line.encode('utf-8'))

    def _handle_openai(self, prompt: str, stream: bool, max_tokens: int):
        t0 = time.time()
        tokens_text = []
        for piece in GLOBAL_ENGINE.generate(prompt, max_tokens=max_tokens):
            tokens_text.append(piece)
        full_text = "".join(tokens_text)
        self._send_json(200, {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": GLOBAL_MODEL_NAME,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": full_text},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 16,
                "completion_tokens": len(tokens_text),
                "total_tokens": 16 + len(tokens_text)
            }
        })


def run_server(model_path: str, host: str = "0.0.0.0", port: int = 11434, n_gpu_layers: int = 0):
    global GLOBAL_ENGINE, GLOBAL_MODEL_NAME
    print("=" * 70)
    print("  SWEEP INFERENCE SERVER (Ollama & OpenAI Drop-In Compatible)")
    print(f"  Model Path:    {model_path}")
    print(f"  GPU Offload:   {n_gpu_layers} layers")
    print(f"  Server URL:    http://{host}:{port}")
    print("=" * 70)

    GLOBAL_MODEL_NAME = os.path.basename(model_path).replace(".gguf", "")
    GLOBAL_ENGINE = SweepEngine(model_path, n_gpu_layers=n_gpu_layers)

    server = ThreadedHTTPServer((host, port), SweepAPIHandler)
    print(f"\n[READY] Server listening on http://{host}:{port}")
    print("  Compatible endpoints:")
    print("    POST /api/generate           (Ollama API)")
    print("    POST /api/chat               (Ollama API)")
    print("    POST /v1/chat/completions    (OpenAI API)")
    print("    GET  /api/tags               (Model Catalog)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        GLOBAL_ENGINE.close()
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="SWEEP Ollama/OpenAI Compatible Server")
    parser.add_argument("--model", type=str, required=True, help="Path to GGUF model")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host binding")
    parser.add_argument("--port", type=int, default=11434, help="Port (11434 for Ollama drop-in)")
    parser.add_argument("--gpu-layers", "-ngl", type=int, default=0, help="Number of GPU layers (0 for CPU, 99 for full GPU)")
    args = parser.parse_args()

    run_server(args.model, host=args.host, port=args.port, n_gpu_layers=args.gpu_layers)

if __name__ == "__main__":
    main()
