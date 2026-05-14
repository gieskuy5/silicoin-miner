/*
 * keccak_native.c — Multi-threaded keccak256 mining kernel for Silicoin
 * Compile: gcc -O3 -march=native -mavx2 -funroll-loops -lpthread -o keccak_native.so -shared -fPIC keccak_native.c
 * 
 * Uses pthreads to split work across all CPU cores at C level
 * (avoids Python multiprocessing overhead)
 */

#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

// Keccak-f[1600] round constants
static const uint64_t RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL
};

static const int ROTC[24] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
    27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44
};

static const int PILN[24] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1
};

#define ROTL64(x, y) (((x) << (y)) | ((x) >> (64 - (y))))

static inline void keccak_f1600(uint64_t state[25]) {
    uint64_t t, bc[5];
    int round, i, j;
    
    for (round = 0; round < 24; round++) {
        // Theta - fully unrolled
        bc[0] = state[0] ^ state[5] ^ state[10] ^ state[15] ^ state[20];
        bc[1] = state[1] ^ state[6] ^ state[11] ^ state[16] ^ state[21];
        bc[2] = state[2] ^ state[7] ^ state[12] ^ state[17] ^ state[22];
        bc[3] = state[3] ^ state[8] ^ state[13] ^ state[18] ^ state[23];
        bc[4] = state[4] ^ state[9] ^ state[14] ^ state[19] ^ state[24];
        
        t = bc[4] ^ ROTL64(bc[1], 1);
        state[0] ^= t; state[5] ^= t; state[10] ^= t; state[15] ^= t; state[20] ^= t;
        t = bc[0] ^ ROTL64(bc[2], 1);
        state[1] ^= t; state[6] ^= t; state[11] ^= t; state[16] ^= t; state[21] ^= t;
        t = bc[1] ^ ROTL64(bc[3], 1);
        state[2] ^= t; state[7] ^= t; state[12] ^= t; state[17] ^= t; state[22] ^= t;
        t = bc[2] ^ ROTL64(bc[4], 1);
        state[3] ^= t; state[8] ^= t; state[13] ^= t; state[18] ^= t; state[23] ^= t;
        t = bc[3] ^ ROTL64(bc[0], 1);
        state[4] ^= t; state[9] ^= t; state[14] ^= t; state[19] ^= t; state[24] ^= t;
        
        // Rho Pi
        t = state[1];
        for (i = 0; i < 24; i++) {
            j = PILN[i];
            bc[0] = state[j];
            state[j] = ROTL64(t, ROTC[i]);
            t = bc[0];
        }
        
        // Chi - unrolled
        for (j = 0; j < 25; j += 5) {
            bc[0] = state[j];
            bc[1] = state[j + 1];
            bc[2] = state[j + 2];
            bc[3] = state[j + 3];
            bc[4] = state[j + 4];
            state[j]     ^= (~bc[1]) & bc[2];
            state[j + 1] ^= (~bc[2]) & bc[3];
            state[j + 2] ^= (~bc[3]) & bc[4];
            state[j + 3] ^= (~bc[4]) & bc[0];
            state[j + 4] ^= (~bc[0]) & bc[1];
        }
        
        // Iota
        state[0] ^= RC[round];
    }
}

static inline void keccak256(const uint8_t *input, int inlen, uint8_t *output) {
    uint64_t state[25];
    uint8_t temp[136];
    
    memset(state, 0, 200);
    memset(temp, 0, 136);
    memcpy(temp, input, inlen);
    temp[inlen] = 0x01;
    temp[135] = 0x80;
    
    const uint64_t *blocks = (const uint64_t *)temp;
    state[0]  = blocks[0];
    state[1]  = blocks[1];
    state[2]  = blocks[2];
    state[3]  = blocks[3];
    state[4]  = blocks[4];
    state[5]  = blocks[5];
    state[6]  = blocks[6];
    state[7]  = blocks[7];
    state[8]  = blocks[8];
    state[9]  = blocks[9];
    state[10] = blocks[10];
    state[11] = blocks[11];
    state[12] = blocks[12];
    state[13] = blocks[13];
    state[14] = blocks[14];
    state[15] = blocks[15];
    state[16] = blocks[16];
    
    keccak_f1600(state);
    memcpy(output, state, 32);
}

/* Single-threaded batch (exported for backward compat) */
int64_t mine_batch(const uint8_t *prefix, const uint8_t *target, 
                   uint64_t start_nonce, uint64_t batch_size) {
    uint8_t input[84];
    uint8_t hash[32];
    uint64_t end = start_nonce + batch_size;
    
    memcpy(input, prefix, 52);
    memset(input + 52, 0, 24);
    
    for (uint64_t n = start_nonce; n < end; n++) {
        input[76] = (uint8_t)(n >> 56);
        input[77] = (uint8_t)(n >> 48);
        input[78] = (uint8_t)(n >> 40);
        input[79] = (uint8_t)(n >> 32);
        input[80] = (uint8_t)(n >> 24);
        input[81] = (uint8_t)(n >> 16);
        input[82] = (uint8_t)(n >> 8);
        input[83] = (uint8_t)(n);
        
        keccak256(input, 84, hash);
        
        // Early exit comparison
        if (hash[0] > target[0]) continue;
        if (hash[0] < target[0]) return (int64_t)n;
        if (hash[1] > target[1]) continue;
        if (hash[1] < target[1]) return (int64_t)n;
        if (hash[2] > target[2]) continue;
        if (hash[2] < target[2]) return (int64_t)n;
        if (hash[3] > target[3]) continue;
        if (hash[3] < target[3]) return (int64_t)n;
        
        // Full comparison for remaining bytes
        int less = 0;
        for (int i = 4; i < 32; i++) {
            if (hash[i] < target[i]) { less = 1; break; }
            if (hash[i] > target[i]) { break; }
        }
        if (less || (memcmp(hash, target, 32) == 0)) return (int64_t)n;
    }
    
    return -1;
}

/* ============================================================
 * MULTI-THREADED MINING
 * ============================================================ */

typedef struct {
    const uint8_t *prefix;
    const uint8_t *target;
    uint64_t start_nonce;
    uint64_t batch_size;
    volatile int64_t *result;      // shared result
    volatile int *found;           // shared flag
} thread_args_t;

static void *mine_thread(void *arg) {
    thread_args_t *args = (thread_args_t *)arg;
    uint8_t input[84];
    uint8_t hash[32];
    uint64_t end = args->start_nonce + args->batch_size;
    
    memcpy(input, args->prefix, 52);
    memset(input + 52, 0, 24);
    
    for (uint64_t n = args->start_nonce; n < end; n++) {
        // Check if another thread found it
        if (*args->found) return NULL;
        
        input[76] = (uint8_t)(n >> 56);
        input[77] = (uint8_t)(n >> 48);
        input[78] = (uint8_t)(n >> 40);
        input[79] = (uint8_t)(n >> 32);
        input[80] = (uint8_t)(n >> 24);
        input[81] = (uint8_t)(n >> 16);
        input[82] = (uint8_t)(n >> 8);
        input[83] = (uint8_t)(n);
        
        keccak256(input, 84, hash);
        
        if (hash[0] > args->target[0]) continue;
        if (hash[0] < args->target[0]) goto found;
        if (hash[1] > args->target[1]) continue;
        if (hash[1] < args->target[1]) goto found;
        if (hash[2] > args->target[2]) continue;
        if (hash[2] < args->target[2]) goto found;
        if (hash[3] > args->target[3]) continue;
        if (hash[3] < args->target[3]) goto found;
        
        {
            int less = 0;
            for (int i = 4; i < 32; i++) {
                if (hash[i] < args->target[i]) { less = 1; break; }
                if (hash[i] > args->target[i]) { break; }
            }
            if (less) goto found;
        }
        continue;
        
    found:
        *args->result = (int64_t)n;
        *args->found = 1;
        return NULL;
    }
    
    return NULL;
}

/*
 * mine_batch_threaded: Split work across N threads at C level
 * Returns found nonce or -1
 */
int64_t mine_batch_threaded(const uint8_t *prefix, const uint8_t *target,
                            uint64_t start_nonce, uint64_t batch_size, int num_threads) {
    if (num_threads <= 0) num_threads = 2;
    if (num_threads > 16) num_threads = 16;
    
    pthread_t threads[16];
    thread_args_t args[16];
    volatile int64_t result = -1;
    volatile int found = 0;
    
    uint64_t chunk = batch_size / num_threads;
    
    for (int i = 0; i < num_threads; i++) {
        args[i].prefix = prefix;
        args[i].target = target;
        args[i].start_nonce = start_nonce + (i * chunk);
        args[i].batch_size = (i == num_threads - 1) ? (batch_size - i * chunk) : chunk;
        args[i].result = &result;
        args[i].found = &found;
        
        pthread_create(&threads[i], NULL, mine_thread, &args[i]);
    }
    
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    return result;
}
