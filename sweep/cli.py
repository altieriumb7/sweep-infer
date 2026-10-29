"""
SWEEP Command-Line Interface (CLI)
Author: Umberto Altieri
License: MIT
"""

import argparse
import sys
import os

from .engine import SweepEngine

def cmd_chat(args):
    print("=" * 65)
    print("  SWEEP Interactive Terminal Chat")
    print(f"  Model: {args.model}")
    print(f"  GPU Layers: {args.gpu_layers}")
    print("  Type 'exit' or 'quit' to terminate.")
    print("=" * 65)

    engine = SweepEngine(args.model, n_gpu_layers=args.gpu_layers)
    history = "<|begin_of_text|>"

    while True:
        try:
            user_input = input("\nUser > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                break

            prompt = history + f"<|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            print("\nSWEEP > ", end="", flush=True)

            bot_response = []
            for token_piece in engine.generate(prompt, max_tokens=args.max_tokens):
                print(token_piece, end="", flush=True)
                bot_response.append(token_piece)
            print()

            history = prompt + "".join(bot_response) + "<|eot_id|>"
        except KeyboardInterrupt:
            print("\nExiting chat...")
            break

    engine.close()

def cmd_serve(args):
    from .server import run_server
    run_server(args.model, host=args.host, port=args.port, n_gpu_layers=args.gpu_layers)

def main():
    parser = argparse.ArgumentParser(prog="sweep", description="SWEEP: High-Performance Inference Engine for 7B/8B LLMs")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Chat subcommand
    p_chat = subparsers.add_parser("chat", help="Interactive terminal chat")
    p_chat.add_argument("--model", "-m", type=str, required=True, help="Path to GGUF model")
    p_chat.add_argument("--gpu-layers", "-ngl", type=int, default=0, help="GPU layers (0 for CPU, 99 for GPU)")
    p_chat.add_argument("--max-tokens", "-n", type=int, default=256, help="Max tokens per turn")
    p_chat.set_defaults(func=cmd_chat)

    # Serve subcommand
    p_serve = subparsers.add_parser("serve", help="Launch Ollama/OpenAI drop-in server")
    p_serve.add_argument("--model", "-m", type=str, required=True, help="Path to GGUF model")
    p_serve.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    p_serve.add_argument("--port", "-p", type=int, default=11434, help="Port (default: 11434 for Ollama)")
    p_serve.add_argument("--gpu-layers", "-ngl", type=int, default=0, help="GPU layers (0 for CPU, 99 for GPU)")
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
