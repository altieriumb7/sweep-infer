import os
import sys
import time
import queue
import threading
import psutil
import requests
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

SNAPSHOT_7B = r"C:\Users\Umberto\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28"
SNAPSHOT_05B = r"C:\Users\Umberto\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Page config
st.set_page_config(
    page_title="SynapticFold v2 — Neural Testing Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #4A90E2, #9013FE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 10px 14px;
        border-left: 4px solid #4A90E2;
        margin-bottom: 10px;
    }
    .metric-val {
        font-size: 1.3rem;
        font-weight: 700;
        color: #212529;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #6c757d;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HARDWARE MONITORING
# ============================================================================
def get_hardware_stats():
    process = psutil.Process()
    rss_mb = process.memory_info().rss / (1024 * 1024)
    rss_gb = rss_mb / 1024
    total_ram_gb = psutil.virtual_memory().total / (1024**3)
    available_ram_gb = psutil.virtual_memory().available / (1024**3)
    cpu_percent = psutil.cpu_percent(interval=None)
    threads = torch.get_num_threads()
    return {
        "rss_mb": rss_mb,
        "rss_gb": rss_gb,
        "total_ram_gb": total_ram_gb,
        "available_ram_gb": available_ram_gb,
        "cpu_percent": cpu_percent,
        "threads": threads,
        "limit_gb": 5.18,
        "within_limit": rss_gb <= 5.18,
    }


# ============================================================================
# MODEL CACHING & LOADERS
# ============================================================================
@st.cache_resource(show_spinner="Caricamento Tokenizer Qwen2.5...")
def get_tokenizer(model_path: str):
    return AutoTokenizer.from_pretrained(model_path, local_files_only=True)


@st.cache_resource(show_spinner="Inizializzazione Modello...")
def get_model(model_choice: str):
    import gc
    gc.collect()

    if model_choice == "🧠 Synaptic Dual-Brain: Talamo Riflesso + Neocorteccia (Consigliato)":
        from synaptic_fold.synaptic_dual_brain import SynapticDualBrain
        brain = SynapticDualBrain(thalamus_path=SNAPSHOT_05B, neocortex_path=SNAPSHOT_7B)
        return {"type": "dual_brain", "brain": brain}

    elif model_choice == "Synaptic 7B: Approccio B (Colonna Corticale BFloat16)":
        from synaptic_fold.synaptic_cortical_column_7b import SynapticCorticalColumn7B, CorticalColumn7BConfig
        cfg = CorticalColumn7BConfig(dtype=torch.bfloat16, device="cpu")
        model = SynapticCorticalColumn7B.from_pretrained(SNAPSHOT_7B, config=cfg)
        tokenizer = get_tokenizer(SNAPSHOT_7B)
        return {"type": "synaptic_column", "model": model, "tokenizer": tokenizer}

    elif model_choice == "Qwen2.5-0.5B Reflex Brain (BFloat16 Puro - Fast)":
        model = AutoModelForCausalLM.from_pretrained(
            SNAPSHOT_05B,
            dtype=torch.bfloat16,
            device_map="cpu",
            local_files_only=True,
        )
        tokenizer = get_tokenizer(SNAPSHOT_05B)
        return {"type": "hf_causal", "model": model, "tokenizer": tokenizer}

    elif model_choice == "Ollama: Llama-3 8B (Q4_0 Local Server)":
        return {"type": "ollama", "model_name": "llama3:latest"}

    elif model_choice == "Synaptic 7B: Full 28 Layer (INT4 Native)":
        from synaptic_fold.synaptic_full28_model import SynapticFull28Model
        model = SynapticFull28Model.from_pretrained(SNAPSHOT_7B)
        tokenizer = get_tokenizer(SNAPSHOT_7B)
        return {"type": "synaptic_full28", "model": model, "tokenizer": tokenizer}

    raise ValueError(f"Modello non supportato: {model_choice}")


# ============================================================================
# STREAMING GENERATORS
# ============================================================================
def stream_synaptic(model_obj, prompt: str, max_tokens: int, temp: float, rep_penalty: float):
    model = model_obj["model"]
    tokenizer = model_obj["tokenizer"]

    msgs = [{"role": "user", "content": prompt}]
    fmt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(fmt, return_tensors="pt")["input_ids"]

    q = queue.Queue()
    stop_event = threading.Event()
    stats = {"ttft": None, "t0": time.perf_counter(), "count": 0}

    def on_token(tok_id: int):
        if stats["ttft"] is None:
            stats["ttft"] = time.perf_counter() - stats["t0"]
        stats["count"] += 1
        tok_str = tokenizer.decode([tok_id], skip_special_tokens=True)
        q.put(tok_str)

    def worker():
        try:
            with torch.no_grad():
                model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temp,
                    top_k=1 if temp <= 0.0 else 50,
                    token_callback=on_token,
                )
        except Exception as e:
            q.put(f"\n[ERRORE GENERAZIONE: {e}]")
        finally:
            stop_event.set()
            q.put(None)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    while True:
        try:
            item = q.get(timeout=0.05)
            if item is None:
                break
            yield item
        except queue.Empty:
            if stop_event.is_set() and q.empty():
                break

    stats["total_time"] = time.perf_counter() - stats["t0"]
    st.session_state["last_stats"] = stats


def stream_hf(model_obj, prompt: str, max_tokens: int, temp: float, rep_penalty: float):
    model = model_obj["model"]
    tokenizer = model_obj["tokenizer"]

    msgs = [{"role": "user", "content": prompt}]
    fmt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(fmt, return_tensors="pt")["input_ids"]

    from transformers import TextIteratorStreamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=max_tokens,
        temperature=max(temp, 1e-4) if temp > 0 else 0.0,
        do_sample=temp > 0,
        repetition_penalty=rep_penalty,
    )

    t0 = time.perf_counter()
    stats = {"ttft": None, "t0": t0, "count": 0}

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
    thread.start()

    for text in streamer:
        if stats["ttft"] is None:
            stats["ttft"] = time.perf_counter() - t0
        stats["count"] += 1
        yield text

    stats["total_time"] = time.perf_counter() - t0
    st.session_state["last_stats"] = stats


def stream_dual_brain(brain_obj, prompt: str, max_tokens: int, temp: float, rep_penalty: float):
    brain = brain_obj["brain"]
    t0 = time.perf_counter()
    stats = {"ttft": None, "t0": t0, "count": 0}
    for chunk in brain.generate_stream(prompt, max_new_tokens=max_tokens, temperature=temp, repetition_penalty=rep_penalty):
        if stats["ttft"] is None and chunk:
            stats["ttft"] = time.perf_counter() - t0
        stats["count"] += 1
        yield chunk
    stats["total_time"] = time.perf_counter() - t0
    st.session_state["last_stats"] = stats


def stream_ollama(model_name: str, prompt: str):
    t0 = time.perf_counter()
    stats = {"ttft": None, "t0": t0, "count": 0}

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": model_name, "prompt": prompt, "stream": True},
            stream=True,
            timeout=120,
        )
        for line in resp.iter_lines():
            if line:
                import json
                chunk = json.loads(line.decode("utf-8"))
                tok = chunk.get("response", "")
                if stats["ttft"] is None and tok:
                    stats["ttft"] = time.perf_counter() - t0
                stats["count"] += 1
                yield tok
                if chunk.get("done", False):
                    break
    except Exception as e:
        yield f"\n[Impossibile contattare Ollama ({OLLAMA_URL}): {e}]"

    stats["total_time"] = time.perf_counter() - t0
    st.session_state["last_stats"] = stats


# ============================================================================
# SIDEBAR UI
# ============================================================================
with st.sidebar:
    st.title("⚙️ Configurazione")

    # Hardware Status Widget
    hw = get_hardware_stats()
    st.markdown("### 🖥️ Hardware Live Monitor")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("RAM Processo", f"{hw['rss_gb']:.2f} GB")
    with col2:
        status_color = "🟢" if hw["within_limit"] else "🔴"
        st.metric("Tetto 5.18 GB", f"{status_color} {'OK' if hw['within_limit'] else 'OVER'}")

    st.progress(min(hw['rss_gb'] / hw['limit_gb'], 1.0), text=f"RAM: {hw['rss_mb']:.0f} MB / 5304 MB")
    st.caption(f"CPU: AMD Ryzen 7 7730U ({hw['threads']} Threads) | Uso CPU: {hw['cpu_percent']}%")

    st.markdown("---")

    # Model Selection (Only the recommended, verified working engine)
    st.markdown("### 🧠 Motore Cognitivo Attivo")
    st.info("✅ **Synaptic Dual-Brain (Talamo Riflesso + Neocorteccia)**\n\nAttualmente attivo in modalità BFloat16 pura. Offre prefill istantaneo (<50 ms), generazione a 15 tok/s e rispetto rigoroso del limite di RAM (700 MB / 5.18 GB).")
    model_choice = "🧠 Synaptic Dual-Brain: Talamo Riflesso + Neocorteccia (Consigliato)"

    st.markdown("---")
    st.markdown("### 🎛️ Iperparametri di Generazione")
    max_tokens = st.slider("Max Nuovi Token:", min_value=10, max_value=250, value=75, step=5)
    temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    rep_penalty = st.slider("Repetition Penalty:", min_value=1.0, max_value=1.5, value=1.15, step=0.05)

    if st.button("🗑️ Cancella Cronologia Chat", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()


# ============================================================================
# MAIN CHAT AREA
# ============================================================================
st.markdown('<div class="main-header">🧠 SynapticFold v2: Neural Chat Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Interfaccia interattiva di test per architetture neuronali bio-ispirate su CPU locale</div>', unsafe_allow_html=True)

# Preset Quick Test Buttons
st.markdown("**Test Rapidi Predefiniti:**")
p_cols = st.columns(3)
preset_clicked = None
with p_cols[0]:
    if st.button("👋 1. Saluto di Sanità", use_container_width=True):
        preset_clicked = "Ciao! Chi sei e qual è il tuo ruolo?"
with p_cols[1]:
    if st.button("📦 2. Puzzle delle 3 Scatole", use_container_width=True):
        preset_clicked = (
            "Ci sono tre scatole: Rossa, Blu e Verde.\n"
            "1. Metto una mela nella scatola Rossa.\n"
            "2. Sposto il contenuto della scatola Rossa nella scatola Blu.\n"
            "3. Scambio le etichette delle scatole Blu e Verde (quella che si chiamava Blu ora si chiama Verde e viceversa).\n"
            "4. Prendo l'oggetto che si trova ora nella scatola chiamata 'Blu' e lo metto nella scatola Rossa.\n"
            "Dove si trova la mela adesso? Spiega il ragionamento passo per passo."
        )
with p_cols[2]:
    if st.button("🐺 3. Indovinello del Fiume", use_container_width=True):
        preset_clicked = (
            "Un contadino deve attraversare un fiume con un cavolo, una pecora e un lupo. "
            "La sua barca può portare solo lui e un oggetto alla volta. "
            "Tuttavia, il contadino ha una corda con cui ha legato saldamente il lupo a un albero sulla riva iniziale, "
            "rendendolo inoffensivo finché resta lì. "
            "Descrivi la sequenza di viaggi più rapida per portare tutto dall'altra parte."
        )

# Initialize messages
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "stats" in msg and msg["stats"]:
            s = msg["stats"]
            ttft_str = f"{s['ttft']:.2f} s" if s.get("ttft") is not None else "N/A"
            speed = s["count"] / max(s.get("total_time", 1), 1e-4)
            st.caption(f"⚡ TTFT: {ttft_str} | Tempo Totale: {s.get('total_time', 0):.2f} s | Token: {s['count']} ({speed:.2f} tok/s)")

# Input Handling
user_input = st.chat_input("Scrivi un messaggio al modello...") or preset_clicked

if user_input:
    # Append and show user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant response
    with st.chat_message("assistant"):
        st.session_state["last_stats"] = None
        response_text = ""

        try:
            active_model = get_model(model_choice)
            m_type = active_model.get("type")

            if m_type == "dual_brain":
                stream = stream_dual_brain(active_model, user_input, max_tokens, temperature, rep_penalty)
            elif m_type == "synaptic_column" or m_type == "synaptic_full28":
                stream = stream_synaptic(active_model, user_input, max_tokens, temperature, rep_penalty)
            elif m_type == "hf_causal":
                stream = stream_hf(active_model, user_input, max_tokens, temperature, rep_penalty)
            elif m_type == "ollama":
                stream = stream_ollama(active_model["model_name"], user_input)
            else:
                stream = ["Tipo modello non riconosciuto."]

            response_text = st.write_stream(stream)
        except Exception as e:
            response_text = f"⚠️ Errore durante l'esecuzione del modello: {e}"
            st.error(response_text)

        # Show performance stats
        stats = st.session_state.get("last_stats")
        if stats and stats["count"] > 0:
            ttft_str = f"{stats['ttft']:.2f} s" if stats.get("ttft") is not None else "N/A"
            speed = stats["count"] / max(stats.get("total_time", 1), 1e-4)
            st.caption(f"⚡ TTFT: {ttft_str} | Tempo Totale: {stats.get('total_time', 0):.2f} s | Token: {stats['count']} ({speed:.2f} tok/s)")

        st.session_state["messages"].append({
            "role": "assistant",
            "content": response_text,
            "stats": stats,
        })
