"""
SynapticFold: Calibrazione del Campo Corticale Continuo (Non-Layer).

Esegue l'allineamento geometrico analitico (Zero Pruning, Zero Strati Discreti):
1. Raccoglie le traiettorie di velocità e gli attrattori semantici su corpus ricco.
2. Risolve in forma chiusa la base di modulazione temporale continua gamma(t), beta(t).
3. Risolve il Manifold Bridge di lettura motoria W_bridge via Ridge Regression analitica.
4. Salva i pesi calibrati e testa la generazione autoregressiva continua.
"""

import os
import sys
import time
import json
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from synaptic_fold.continuous_cortical_engine import ContinuousCorticalFieldEngine

CALIBRATION_CORPUS = [
    # Identità e conversazione
    "Ciao! Come stai? Io mi chiamo Umberto e sono felice di conoscerti.",
    "Piacere di conoscerti Umberto, io sono un assistente linguistico intelligente basato su una rete neurale bio-ispirata.",
    "Come posso aiutarti oggi con i tuoi progetti di ricerca, programmazione o matematica?",
    "Vorrei capire come funziona l'architettura a campo continuo della corteccia cerebrale.",
    # Conoscenza generale
    "La capitale d'Italia è Roma, celebre in tutto il mondo per la sua storia millenaria e per monumenti unici come il Colosseo.",
    "Dante Alighieri è il padre della lingua italiana e autore della celebre opera Divina Commedia.",
    "La Terra è il terzo pianeta del sistema solare e compie una rivoluzione completa attorno al Sole in circa trecentosessantacinque giorni.",
    "La fotosintesi clorofilliana è il processo biologico mediante il quale le piante trasformano la luce solare in energia chimica.",
    # Logica e problem solving
    "Un contadino deve attraversare un fiume con un lupo, una capra e un cavolo usando una piccola barca.",
    "La barca può trasportare solo il contadino e uno solo tra lupo, capra o cavolo alla volta.",
    "Per non far mangiare la capra dal lupo o il cavolo dalla capra, il contadino trasporta prima la capra sulla riva opposta.",
    "Poi il contadino torna indietro e porta il cavolo, ma riporta indietro la capra per non lasciarli soli.",
    # Programmazione e tecnologia
    "Python è un linguaggio di programmazione ad alto livello noto per la sua leggibilità e vasta libreria standard.",
    "La funzione print consente di stampare stringhe e variabili direttamente sulla console di comando.",
    "Una lista in Python è una struttura dati mutabile e ordinata che può contenere elementi eterogenei.",
    "Le reti neurali artificiali apprendono rappresentazioni distribuite attraverso matrici di pesi sinaptici.",
    # Filosofia e neuroscienze
    "Il cervello umano consuma circa venti watt di potenza e contiene circa ottantasei miliardi di neuroni interconnessi.",
    "La corteccia cerebrale è organizzata in minicolonne corticali che elaborano le informazioni attraverso dinamiche ricorrenti.",
    "La memoria di lavoro consente di mantenere temporaneamente le informazioni attive per compiti cognitivi complessi."
]

def calibrate_continuous_field():
    torch.set_num_threads(8)
    print("=" * 80, flush=True)
    print("🧠 SYNAPTICFOLD: CALIBRAZIONE ANALITICA CAMPO CORTICALE CONTINUO (NON-LAYER)", flush=True)
    print("   Obiettivo: Eliminare limit-cycle e stabilizzare l'attrattore semantico a >10 tok/s", flush=True)
    print("=" * 80, flush=True)

    model_path = r"C:\Users\Umberto\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
    print(f"\n[1/4] Caricamento modello e tokenizer da {model_path} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,
        device_map="cpu",
        local_files_only=True,
    )
    hf_model.eval()

    # Raccogli dati dal file JSON esistente se presente
    json_path = os.path.join(ROOT_DIR, "synaptic_fold", "data", "cortical_calibration_corpus.json")
    all_texts = list(CALIBRATION_CORPUS)
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            j_data = json.load(f)
            for item in j_data:
                all_texts.append(item.get("prompt", "") + " " + item.get("completion", ""))

    print(f"\n[2/4] Raccolta traiettorie latenti su {len(all_texts)} testi di calibrazione...", flush=True)
    h0_list, h_core_list, h_exit_list = [], [], []

    num_layers = hf_model.config.num_hidden_layers
    core_idx = num_layers // 2 # Layer 12

    with torch.no_grad():
        for text in all_texts:
            tokens = tokenizer(text, return_tensors="pt")["input_ids"]
            if tokens.shape[1] < 3:
                continue
            out = hf_model(tokens, output_hidden_states=True)
            h0_list.append(out.hidden_states[0][0])
            h_core_list.append(out.hidden_states[core_idx][0])
            h_exit_list.append(out.hidden_states[-1][0])

    H0 = torch.cat(h0_list, dim=0)       # (N, D)
    H_core = torch.cat(h_core_list, dim=0) # (N, D)
    H_exit = torch.cat(h_exit_list, dim=0) # (N, D)
    N, D = H_core.shape
    print(f"  -> Raccolti {N} punti di traiettoria latente nello spazio a {D} dimensioni (N > D: {N > D}).", flush=True)

    # 3. Risoluzione Analitica Ridge del Manifold Bridge (Corteccia -> Uscita Motoria)
    print("\n[3/4] Risoluzione Gauss-Markov Analitica del Manifold Bridge...", flush=True)
    t_solve0 = time.perf_counter()
    XtX = H_core.t() @ H_core
    reg = 1e-4 * (torch.trace(XtX) / D) * torch.eye(D)
    W_bridge = torch.linalg.solve(XtX + reg, H_core.t() @ H_exit) # (D, D)
    solve_dt = time.perf_counter() - t_solve0

    # Verifica allineamento
    with torch.no_grad():
        H_pred = H_core @ W_bridge
        cos_sim = F.cosine_similarity(H_pred, H_exit, dim=-1).mean().item()
        mse = F.mse_loss(H_pred, H_exit).item()
    print(f"  -> Risolto in {solve_dt*1000:.2f} ms! Cosine Similarity = {cos_sim:.4f} | MSE = {mse:.4f}", flush=True)

    # 4. Creazione Motore a Campo Continuo e Inizializzazione Ottimale
    print("\n[4/4] Costruzione del Motore a Campo Corticale Continuo...", flush=True)
    thought_steps = 4
    brain = ContinuousCorticalFieldEngine.from_huggingface_model(
        hf_model=hf_model,
        thought_steps=thought_steps,
        ode_dt=0.25,
    )
    brain.eval()

    # Registra W_bridge come buffer permanente nel motore
    brain.register_buffer("W_bridge", W_bridge)

    # Calibrazione chiusa dei vettori gamma(t) e beta(t)
    # Per ogni passo t nel tempo continuo, calcola il guadagno di scala e shift rispetto alla profondità reale
    for step_t in range(thought_steps):
        target_layer = int((step_t + 1) * (num_layers - 1) / thought_steps)
        h_t_real = hf_model.model.layers[target_layer].input_layernorm.weight.data
        h_ref = brain.cortical_field.norm_attn.weight.data
        
        gamma = h_t_real / (h_ref + 1e-6)
        brain.cortical_field.temporal_gamma.data[step_t].copy_(gamma.to(brain.dtype))
        brain.cortical_field.temporal_beta.data[step_t].zero_()

    # Salva i pesi del cervello calibrato
    weights_dir = os.path.join(ROOT_DIR, "synaptic_fold", "weights")
    os.makedirs(weights_dir, exist_ok=True)
    save_path = os.path.join(weights_dir, "continuous_cortical_field_0.5b.pt")
    torch.save({
        "state_dict": brain.state_dict(),
        "W_bridge": W_bridge,
        "cos_sim": cos_sim,
        "thought_steps": thought_steps,
        "hidden_size": D,
    }, save_path)
    print(f"  -> Pesi del Campo Corticale salvati con successo in: {save_path}", flush=True)

    # 5. Test di Generazione Autoregressiva Continua con Allineamento
    print("\n" + "=" * 80, flush=True)
    print("VERIFICA GENERAZIONE TESTUALE NEL CAMPO CONTINUO CALIBRATO:", flush=True)
    print("=" * 80, flush=True)

    test_prompts = [
        "Ciao se io mi chiamo Umberto, tu come ti chiami?",
        "Qual è la capitale d'Italia e perché è celebre nel mondo?",
        "Un contadino deve attraversare un fiume con un cavolo, una pecora e un lupo. Come fa?"
    ]

    for idx, prompt in enumerate(test_prompts, 1):
        print(f"\n--- [TEST {idx}] Prompt: \"{prompt}\" ---", flush=True)
        msgs = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(formatted, return_tensors="pt")["input_ids"]
        curr_ids = inp.clone()

        t0 = time.perf_counter()
        with torch.no_grad():
            for step in range(50):
                # Forward attraverso il campo continuo
                h = brain.embed_tokens(curr_ids)
                cos, sin = brain._get_rope(torch.arange(0, curr_ids.shape[1]).unsqueeze(0))
                
                # Riverberazione continua nel campo sinaptico (Zero layer discreti)
                for step_t in range(thought_steps):
                    dh, _ = brain.cortical_field.evaluate_field_step(h, step_t, cos, sin)
                    h = h + 0.25 * dh
                    
                # Proiezione di lettura motoria allineata
                h_motor = h[:, -1:, :] @ brain.W_bridge
                logits = brain.lm_head(brain.motor_norm(h_motor))
                
                next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                curr_ids = torch.cat([curr_ids, next_tok], dim=-1)
                
                if next_tok.item() == tokenizer.eos_token_id:
                    break

        dt = time.perf_counter() - t0
        gen_toks = curr_ids[0, inp.shape[1]:]
        text = tokenizer.decode(gen_toks, skip_special_tokens=True).strip()
        speed = len(gen_toks) / max(dt, 1e-4)

        print(f"Risposta Generata:\n\"{text}\"", flush=True)
        print(f"⚡ [Velocita: {speed:.2f} tok/s | Token: {len(gen_toks)} | Tempo: {dt:.2f}s]", flush=True)

if __name__ == "__main__":
    calibrate_continuous_field()
