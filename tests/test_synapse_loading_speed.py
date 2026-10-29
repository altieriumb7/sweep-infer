import os
import sys
import time
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from synaptic_fold.synaptic_cortical_engine import SynapticCorticalEngine

synapse_file = os.path.join(ROOT_DIR, "models", "qwen2.5_7b_cortical.synapse")
if not os.path.exists(synapse_file):
    print("Synapse file not found yet.")
    sys.exit(0)

print(f"Loading SynapticCorticalEngine directly from native binary ({synapse_file})...")
t0 = time.perf_counter()
model = SynapticCorticalEngine.from_synapse(synapse_file)
dt = time.perf_counter() - t0
print(f"Successfully loaded 28 layers in {dt:.2f} seconds!")
mem = model.get_memory_breakdown()
print(f"RAM: {mem['model_ram_mb']:.1f} MB ({mem['model_ram_gb']:.2f} GB) | RSS: {mem['rss_mb']:.1f} MB")
