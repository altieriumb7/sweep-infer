/*
 * SWEEP: Single-pass Wavefront Entropy-gated Execution Pipeline
 * Native C Application Programming Interface (API)
 * 
 * Target: Complete 7B/8B Parameter Models (Meta-Llama-3-8B / Qwen2.5-7B)
 * Author: Umberto Altieri
 * License: MIT
 */

#ifndef SWEEP_H
#define SWEEP_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#if defined(_WIN32)
  #if defined(SWEEP_BUILD_DLL)
    #define SWEEP_API __declspec(dllexport)
  #else
    #define SWEEP_API __declspec(dllimport)
  #endif
#else
  #define SWEEP_API __attribute__((visibility("default")))
#endif

/* Data Types */
typedef int32_t sweep_token_t;
typedef struct sweep_engine sweep_engine_t;

/* Engine Configuration */
typedef struct {
    const char* model_path;          /* Path to GGUF model file */
    int32_t     n_gpu_layers;        /* 0 for CPU-only, 99 for full GPU offload */
    int32_t     n_threads;           /* CPU computation threads (default: 8) */
    int32_t     n_ctx;               /* Maximum context length (e.g. 2048, 4096) */
    int32_t     wavefront_k;         /* Wavefront batch size (default: 2, up to 6) */
    float       entropy_threshold;   /* Shannon entropy gate threshold (default: 0.55) */
    float       margin_threshold;    /* Top-1 vs Top-2 margin threshold (default: 0.60) */
    bool        verbose;             /* Enable detailed engine telemetry logs */
} sweep_config_t;

/* Performance Telemetry */
typedef struct {
    double  tokens_per_second;       /* Overall generation speed */
    double  acceptance_rate;         /* Speculative acceptance rate alpha [0.0 - 1.0] */
    int32_t total_tokens_generated;  /* Total tokens produced */
    int32_t total_wavefront_sweeps;  /* Total DRAM sweeps executed */
    int32_t speculative_accepted;    /* Accepted candidate tokens */
    int32_t speculative_proposed;    /* Proposed candidate tokens */
    double  elapsed_seconds;         /* Total execution time in seconds */
} sweep_stats_t;

/* Callback signature for streaming generation */
typedef bool (*sweep_token_callback_t)(sweep_token_t token, const char* piece, void* user_data);

/*
 * Initialize the default configuration with recommended settings.
 */
SWEEP_API sweep_config_t sweep_default_config(void);

/*
 * Create and initialize a new SWEEP engine instance.
 * Returns NULL on failure.
 */
SWEEP_API sweep_engine_t* sweep_create(const sweep_config_t* config);

/*
 * Free all resources associated with a SWEEP engine instance.
 */
SWEEP_API void sweep_free(sweep_engine_t* engine);

/*
 * Tokenize an input string into token IDs.
 * Returns the number of tokens written into out_tokens, or negative on buffer overflow.
 */
SWEEP_API int32_t sweep_tokenize(
    sweep_engine_t* engine,
    const char* text,
    sweep_token_t* out_tokens,
    int32_t max_tokens
);

/*
 * Convert a token ID into its string representation.
 * Returns the length written into out_piece.
 */
SWEEP_API int32_t sweep_token_to_piece(
    sweep_engine_t* engine,
    sweep_token_t token,
    char* out_piece,
    int32_t max_len
);

/*
 * Execute end-to-end wavefront generation for a prompt.
 * If callback is provided, it is invoked for every verified token as it is generated.
 * Returns the total number of tokens generated, or negative on error.
 */
SWEEP_API int32_t sweep_generate(
    sweep_engine_t* engine,
    const char* prompt,
    int32_t max_new_tokens,
    sweep_token_callback_t callback,
    void* user_data
);

/*
 * Retrieve the performance telemetry and metrics from the last generation run.
 */
SWEEP_API void sweep_get_stats(
    sweep_engine_t* engine,
    sweep_stats_t* out_stats
);

/*
 * Returns the engine version string.
 */
SWEEP_API const char* sweep_version(void);

#ifdef __cplusplus
}
#endif

#endif /* SWEEP_H */
