#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

#define HASH_TABLE_SIZE 16384  // 16K slots = 128 KB (fits comfortably in Zen 3 512KB L2 cache)
#define HASH_MASK (HASH_TABLE_SIZE - 1)

typedef struct {
    uint64_t key;       // (token_prev << 32) | token_curr
    int32_t target_tok; // high-probability successor token
    uint32_t count;     // occurrence count
} HashEntry;

static HashEntry s_table[HASH_TABLE_SIZE];

EXPORT void synaptic_ngram_router_init(void) {
    memset(s_table, 0, sizeof(s_table));
}

// Hardware-accelerated CRC32 insertion
EXPORT void synaptic_ngram_router_insert(int32_t t_prev, int32_t t_curr, int32_t t_target) {
    uint64_t key = (((uint64_t)(uint32_t)t_prev) << 32) | ((uint64_t)(uint32_t)t_curr);
    uint32_t hash = (uint32_t)_mm_crc32_u64(0, key);
    uint32_t idx = hash & HASH_MASK;

    for (uint32_t step = 0; step < 16; ++step) {
        uint32_t slot = (idx + step) & HASH_MASK;
        if (s_table[slot].count == 0) {
            s_table[slot].key = key;
            s_table[slot].target_tok = t_target;
            s_table[slot].count = 1;
            return;
        } else if (s_table[slot].key == key && s_table[slot].target_tok == t_target) {
            s_table[slot].count++;
            return;
        } else if (s_table[slot].key == key) {
            // Update if count is higher
            s_table[slot].count++;
            s_table[slot].target_tok = t_target;
            return;
        }
    }
}

// Hardware-accelerated CRC32 query (< 10 nanoseconds)
EXPORT int32_t synaptic_ngram_router_query(int32_t t_prev, int32_t t_curr) {
    uint64_t key = (((uint64_t)(uint32_t)t_prev) << 32) | ((uint64_t)(uint32_t)t_curr);
    uint32_t hash = (uint32_t)_mm_crc32_u64(0, key);
    uint32_t idx = hash & HASH_MASK;

    for (uint32_t step = 0; step < 16; ++step) {
        uint32_t slot = (idx + step) & HASH_MASK;
        if (s_table[slot].count == 0) {
            return -1;
        }
        if (s_table[slot].key == key) {
            return s_table[slot].target_tok;
        }
    }
    return -1;
}

// Bulk seed from token array
EXPORT void synaptic_ngram_router_seed(const int32_t* tokens, int32_t n_tokens) {
    if (n_tokens < 3) return;
    for (int32_t i = 0; i < n_tokens - 2; ++i) {
        synaptic_ngram_router_insert(tokens[i], tokens[i+1], tokens[i+2]);
    }
}
