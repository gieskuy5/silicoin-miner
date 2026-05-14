/*
 * keccak256.cl — OpenCL GPU kernel for Silicoin mining
 * 
 * Each work-item tries one nonce. Launch millions of work-items
 * to saturate the GPU.
 *
 * Input layout per hash: prefix(52 bytes) + nonce(32 bytes) = 84 bytes
 * Output: if hash < target, write nonce to result buffer
 */

// Keccak-f[1600] round constants
__constant ulong RC[24] = {
    0x0000000000000001UL, 0x0000000000008082UL,
    0x800000000000808aUL, 0x8000000080008000UL,
    0x000000000000808bUL, 0x0000000080000001UL,
    0x8000000080008081UL, 0x8000000000008009UL,
    0x000000000000008aUL, 0x0000000000000088UL,
    0x0000000080008009UL, 0x000000008000000aUL,
    0x000000008000808bUL, 0x800000000000008bUL,
    0x8000000000008089UL, 0x8000000000008003UL,
    0x8000000000008002UL, 0x8000000000000080UL,
    0x000000000000800aUL, 0x800000008000000aUL,
    0x8000000080008081UL, 0x8000000000008080UL,
    0x0000000080000001UL, 0x8000000080008008UL
};

__constant int PILN[24] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1
};

__constant int ROTC[24] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
    27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44
};

#define ROTL64(x, y) (((x) << (y)) | ((x) >> (64 - (y))))

void keccak_f1600(ulong state[25]) {
    ulong t, bc[5];
    
    for (int round = 0; round < 24; round++) {
        // Theta
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
        for (int i = 0; i < 24; i++) {
            int j = PILN[i];
            bc[0] = state[j];
            state[j] = ROTL64(t, ROTC[i]);
            t = bc[0];
        }
        
        // Chi
        for (int j = 0; j < 25; j += 5) {
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

/*
 * Main mining kernel
 * 
 * prefix: 52 bytes (challenge[32] + miner_address[20])
 * target: 32 bytes (big-endian uint256)
 * start_nonce: base nonce (each work-item adds its global_id)
 * result: output buffer [0] = found flag, [1] = nonce_hi, [2] = nonce_lo
 */
__kernel void mine_keccak256(
    __global const uchar *prefix,      // 52 bytes
    __global const uchar *target,      // 32 bytes
    ulong start_nonce,
    __global volatile uint *result      // [found, nonce_hi, nonce_lo]
) {
    // Early exit if another work-item already found a solution
    if (result[0]) return;
    
    ulong nonce = start_nonce + get_global_id(0);
    
    // Build input: prefix(52) + nonce(32) = 84 bytes
    uchar input[136];  // padded to rate
    for (int i = 0; i < 52; i++) input[i] = prefix[i];
    
    // Write nonce as big-endian 32 bytes (first 24 = 0, last 8 = nonce)
    for (int i = 52; i < 76; i++) input[i] = 0;
    input[76] = (uchar)(nonce >> 56);
    input[77] = (uchar)(nonce >> 48);
    input[78] = (uchar)(nonce >> 40);
    input[79] = (uchar)(nonce >> 32);
    input[80] = (uchar)(nonce >> 24);
    input[81] = (uchar)(nonce >> 16);
    input[82] = (uchar)(nonce >> 8);
    input[83] = (uchar)(nonce);
    
    // Keccak padding (rate=136 for keccak256)
    input[84] = 0x01;
    for (int i = 85; i < 135; i++) input[i] = 0;
    input[135] = 0x80;
    
    // Absorb into state
    ulong state[25];
    for (int i = 0; i < 25; i++) state[i] = 0;
    
    // XOR input into state (17 uint64s = 136 bytes)
    for (int i = 0; i < 17; i++) {
        ulong val = 0;
        for (int b = 0; b < 8; b++) {
            val |= ((ulong)input[i * 8 + b]) << (b * 8);
        }
        state[i] = val;
    }
    
    // Permute
    keccak_f1600(state);
    
    // Extract hash (first 32 bytes of state, little-endian → big-endian compare)
    uchar hash[32];
    for (int i = 0; i < 4; i++) {
        for (int b = 0; b < 8; b++) {
            hash[i * 8 + b] = (uchar)(state[i] >> (b * 8));
        }
    }
    
    // Compare hash < target (big-endian, early exit)
    bool less = false;
    for (int i = 0; i < 32; i++) {
        if (hash[i] < target[i]) { less = true; break; }
        if (hash[i] > target[i]) { break; }
    }
    
    if (less) {
        // Atomic write result
        if (atomic_cmpxchg(&result[0], 0u, 1u) == 0u) {
            result[1] = (uint)(nonce >> 32);
            result[2] = (uint)(nonce & 0xFFFFFFFF);
        }
    }
}
