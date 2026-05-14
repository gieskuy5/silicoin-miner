"""
Silicoin CUDA Miner — NVIDIA GPU mining via PyCUDA
Requires: NVIDIA GPU + CUDA toolkit + pycuda

Install:
  pip install pycuda numpy
"""

import os
import time
import numpy as np

try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    PYCUDA_AVAILABLE = True
except ImportError:
    PYCUDA_AVAILABLE = False


CUDA_KERNEL = """
// Keccak-f[1600] round constants
__device__ __constant__ unsigned long long RC[24] = {
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

__device__ __constant__ int ROTC[24] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
    27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44
};

__device__ __constant__ int PILN[24] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1
};

#define ROTL64(x, y) (((x) << (y)) | ((x) >> (64 - (y))))

__device__ void keccak_f1600(unsigned long long state[25]) {
    unsigned long long t, bc[5];
    
    #pragma unroll
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
        #pragma unroll
        for (int i = 0; i < 24; i++) {
            int j = PILN[i];
            bc[0] = state[j];
            state[j] = ROTL64(t, ROTC[i]);
            t = bc[0];
        }
        
        // Chi
        #pragma unroll
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

__device__ void keccak256(const unsigned char *input, int inlen, unsigned char *output) {
    unsigned long long state[25];
    unsigned char temp[136];
    
    // Zero state
    #pragma unroll
    for (int i = 0; i < 25; i++) state[i] = 0;
    
    // Pad input into rate block
    #pragma unroll
    for (int i = 0; i < 136; i++) temp[i] = 0;
    
    for (int i = 0; i < inlen; i++) temp[i] = input[i];
    temp[inlen] = 0x01;
    temp[135] = 0x80;
    
    // XOR into state
    unsigned long long *blocks = (unsigned long long *)temp;
    #pragma unroll
    for (int i = 0; i < 17; i++) state[i] ^= blocks[i];
    
    keccak_f1600(state);
    
    // Squeeze 32 bytes
    unsigned char *out = (unsigned char *)state;
    #pragma unroll
    for (int i = 0; i < 32; i++) output[i] = out[i];
}

// Main mining kernel
// Each thread tries one nonce
__global__ void mine_kernel(
    const unsigned char *prefix,    // 52 bytes (challenge + address)
    const unsigned char *target,    // 32 bytes
    unsigned long long start_nonce,
    unsigned long long *result,     // output: found nonce
    int *found                      // output: flag
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long nonce = start_nonce + (unsigned long long)tid;
    
    // Early exit if another thread found it
    if (*found) return;
    
    // Build input: prefix(52) + nonce(32) = 84 bytes
    unsigned char input[84];
    
    // Copy prefix
    #pragma unroll
    for (int i = 0; i < 52; i++) input[i] = prefix[i];
    
    // Write nonce as big-endian 32 bytes (only last 8 bytes used)
    #pragma unroll
    for (int i = 52; i < 76; i++) input[i] = 0;
    
    input[76] = (unsigned char)(nonce >> 56);
    input[77] = (unsigned char)(nonce >> 48);
    input[78] = (unsigned char)(nonce >> 40);
    input[79] = (unsigned char)(nonce >> 32);
    input[80] = (unsigned char)(nonce >> 24);
    input[81] = (unsigned char)(nonce >> 16);
    input[82] = (unsigned char)(nonce >> 8);
    input[83] = (unsigned char)(nonce);
    
    // Hash
    unsigned char hash[32];
    keccak256(input, 84, hash);
    
    // Compare hash < target (big-endian, early exit)
    for (int i = 0; i < 32; i++) {
        if (hash[i] < target[i]) {
            // Found! Atomically set result
            atomicExch((unsigned long long *)result, nonce);
            atomicExch(found, 1);
            return;
        }
        if (hash[i] > target[i]) return;
    }
    // Equal — also valid
    atomicExch((unsigned long long *)result, nonce);
    atomicExch(found, 1);
}
"""


class CUDAMiner:
    def __init__(self, device_idx=0, dry_run=False):
        if not PYCUDA_AVAILABLE:
            raise RuntimeError("pycuda not installed. Run: pip install pycuda numpy")
        
        # Get device info
        self.device = cuda.Device(device_idx)
        self.device_name = self.device.name()
        self.compute_units = self.device.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)
        
        # Optimal grid size based on GPU
        # Each SM can run multiple blocks; aim for high occupancy
        self.block_size = 256  # threads per block
        self.grid_size = self.compute_units * 32  # blocks
        self.global_work_size = self.block_size * self.grid_size
        
        # Estimate hashrate (~1000 H/s per CUDA core at base clock)
        cuda_cores_per_sm = self._get_cuda_cores_per_sm()
        total_cores = self.compute_units * cuda_cores_per_sm
        self.est_hashrate = total_cores * 800  # conservative estimate
        
        if dry_run:
            return
        
        # Compile kernel
        self.module = SourceModule(CUDA_KERNEL, no_extern_c=True)
        self.kernel = self.module.get_function("mine_kernel")
        
        # Allocate device memory
        self.d_prefix = cuda.mem_alloc(52)
        self.d_target = cuda.mem_alloc(32)
        self.d_result = cuda.mem_alloc(8)
        self.d_found = cuda.mem_alloc(4)
    
    def _get_cuda_cores_per_sm(self):
        """Estimate CUDA cores per SM based on compute capability."""
        cc = self.device.compute_capability()
        major, minor = cc
        # Approximate cores per SM by architecture
        cores_map = {
            (7, 0): 64,   # Volta (V100)
            (7, 5): 64,   # Turing (RTX 2000)
            (8, 0): 64,   # Ampere (A100)
            (8, 6): 128,  # Ampere (RTX 3000)
            (8, 9): 128,  # Ada Lovelace (RTX 4000)
            (9, 0): 128,  # Hopper (H100)
        }
        return cores_map.get((major, minor), 64)
    
    def mine_batch(self, prefix: bytes, target: bytes, start_nonce: int) -> int:
        """
        Mine a batch of nonces on GPU.
        Returns found nonce or -1.
        """
        # Upload prefix and target
        cuda.memcpy_htod(self.d_prefix, np.frombuffer(prefix, dtype=np.uint8))
        cuda.memcpy_htod(self.d_target, np.frombuffer(target, dtype=np.uint8))
        
        # Reset result
        h_result = np.array([0], dtype=np.uint64)
        h_found = np.array([0], dtype=np.int32)
        cuda.memcpy_htod(self.d_result, h_result)
        cuda.memcpy_htod(self.d_found, h_found)
        
        # Launch kernel
        self.kernel(
            self.d_prefix,
            self.d_target,
            np.uint64(start_nonce),
            self.d_result,
            self.d_found,
            block=(self.block_size, 1, 1),
            grid=(self.grid_size, 1)
        )
        
        # Sync and read result
        cuda.Context.synchronize()
        
        cuda.memcpy_dtoh(h_found, self.d_found)
        if h_found[0]:
            cuda.memcpy_dtoh(h_result, self.d_result)
            return int(h_result[0])
        
        return -1
    
    def benchmark(self, duration=3.0) -> float:
        """Run benchmark for `duration` seconds, return H/s."""
        import secrets
        prefix = secrets.token_bytes(52)
        target = (2**200).to_bytes(32, 'big')  # very hard, won't find
        
        cuda.memcpy_htod(self.d_prefix, np.frombuffer(prefix, dtype=np.uint8))
        cuda.memcpy_htod(self.d_target, np.frombuffer(target, dtype=np.uint8))
        
        total_hashes = 0
        start = time.time()
        nonce = 0
        
        while time.time() - start < duration:
            h_result = np.array([0], dtype=np.uint64)
            h_found = np.array([0], dtype=np.int32)
            cuda.memcpy_htod(self.d_result, h_result)
            cuda.memcpy_htod(self.d_found, h_found)
            
            self.kernel(
                self.d_prefix,
                self.d_target,
                np.uint64(nonce),
                self.d_result,
                self.d_found,
                block=(self.block_size, 1, 1),
                grid=(self.grid_size, 1)
            )
            cuda.Context.synchronize()
            
            total_hashes += self.global_work_size
            nonce += self.global_work_size
        
        elapsed = time.time() - start
        return total_hashes / elapsed
