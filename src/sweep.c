/*
 * SWEEP: Single-pass Wavefront Entropy-gated Execution Pipeline
 * Native C Implementation
 * 
 * Author: Umberto Altieri
 * License: MIT
 */

#include "sweep.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/time.h>
#endif

/* Forward declarations for llama.cpp C API */
typedef void* llama_model_p;
typedef void* llama_context_p;
typedef void* llama_vocab_p;
typedef void* llama_memory_p;

typedef struct {
    int32_t n_tokens;
    int32_t* token;
    float* embd;
    int32_t* pos;
    int32_t* n_seq_id;
    int32_t** seq_id;
    int8_t* logits;
} llama_batch_t;

/* llama.cpp function pointers or external symbols */
extern llama_model_p llama_model_load_from_file(const char* path, void* params);
extern void llama_model_free(llama_model_p model);
extern llama_vocab_p llama_model_get_vocab(llama_model_p model);
extern int32_t llama_vocab_n_tokens(llama_vocab_p vocab);
extern llama_context_p llama_init_from_model(llama_model_p model, void* params);
extern void llama_free(llama_context_p ctx);
extern llama_batch_t llama_batch_init(int32_t n_tokens, int32_t embd, int32_t n_seq_max);
extern void llama_batch_free(llama_batch_t batch);
extern int32_t llama_decode(llama_context_p ctx, llama_batch_t batch);
extern float* llama_get_logits(llama_context_p ctx);
extern float* llama_get_logits_ith(llama_context_p ctx, int32_t i);
extern llama_memory_p llama_get_memory(llama_context_p ctx);
extern bool llama_memory_seq_rm(llama_memory_p mem, int32_t seq_id, int32_t p0, int32_t p1);
extern int32_t llama_tokenize(llama_vocab_p vocab, const char* text, int32_t text_len, int32_t* tokens, int32_t max_tokens, bool add_special, bool parse_special);
extern int32_t llama_token_to_piece(llama_vocab_p vocab, int32_t token, char* buf, int32_t max_len, int32_t lstrip, bool special);
extern void* llama_model_default_params(void);
extern void* llama_context_default_params(void);
extern void llama_backend_init(void);

/* Fast timer helper */
static double get_time_sec(void) {
#if defined(_WIN32)
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

/* Internal Engine State */
struct sweep_engine {
    sweep_config_t  config;
    llama_model_p   model;
    llama_vocab_p   vocab;
    int32_t         vocab_size;
    sweep_stats_t   last_stats;
};

SWEEP_API sweep_config_t sweep_default_config(void) {
    sweep_config_t cfg;
    cfg.model_path = NULL;
    cfg.n_gpu_layers = 0;       /* CPU default, set 99 for GPU */
    cfg.n_threads = 8;
    cfg.n_ctx = 2048;
    cfg.wavefront_k = 2;
    cfg.entropy_threshold = 0.55f;
    cfg.margin_threshold = 0.60f;
    cfg.verbose = false;
    return cfg;
}

SWEEP_API const char* sweep_version(void) {
    return "SWEEP 2.0.0 (Wavefront Entropy-Gated Speculative Engine)";
}

SWEEP_API sweep_engine_t* sweep_create(const sweep_config_t* config) {
    if (!config || !config->model_path) return NULL;

    sweep_engine_t* engine = (sweep_engine_t*)calloc(1, sizeof(sweep_engine_t));
    if (!engine) return NULL;

    engine->config = *config;
    memset(&engine->last_stats, 0, sizeof(sweep_stats_t));

    return engine;
}

SWEEP_API void sweep_free(sweep_engine_t* engine) {
    if (!engine) return;
    free(engine);
}

SWEEP_API void sweep_get_stats(sweep_engine_t* engine, sweep_stats_t* out_stats) {
    if (!engine || !out_stats) return;
    *out_stats = engine->last_stats;
}
