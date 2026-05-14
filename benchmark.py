"""
Benchmark your CPU hashrate for Silicoin mining.
Run this before mining to see what speed you'll get.

Usage: python3 benchmark.py
"""

import time
import secrets
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("⛏️  SILICOIN MINER — BENCHMARK")
print("=" * 60)
print()

CPU_COUNT = os.cpu_count() or 1
print(f"CPU cores detected: {CPU_COUNT}")
print()

# Test data
prefix = secrets.token_bytes(52)  # challenge(32) + address(20)
hard_target = (2**200).to_bytes(32, 'big')  # very hard, won't find

# Try native C
try:
    from native_hasher import mine_batch_native, HAS_NATIVE, get_hashrate_estimate
    print(f"Native C library: {'✅ Loaded' if HAS_NATIVE else '❌ Not compiled'}")
except ImportError:
    HAS_NATIVE = False
    print("Native C library: ❌ Not found (compile with gcc first)")

print()
print("-" * 60)

BATCH = 2_000_000

if HAS_NATIVE:
    # Test single thread
    print(f"\n[1 thread] Testing {BATCH:,} hashes...")
    start = time.time()
    mine_batch_native(prefix, hard_target, 0, BATCH, threads=1)
    t1 = time.time() - start
    rate1 = BATCH / t1
    print(f"  → {t1:.3f}s | {rate1:,.0f} H/s")
    
    # Test multi-thread (up to 16)
    max_threads = min(CPU_COUNT, 16)
    best_rate = rate1
    best_threads = 1
    
    for t in [2, 4, 8, 16]:
        if t > max_threads:
            break
        print(f"\n[{t} threads] Testing {BATCH:,} hashes...")
        start = time.time()
        mine_batch_native(prefix, hard_target, 0, BATCH, threads=t)
        elapsed = time.time() - start
        rate = BATCH / elapsed
        speedup = rate / rate1
        print(f"  → {elapsed:.3f}s | {rate:,.0f} H/s | {speedup:.2f}x speedup")
        
        if rate > best_rate:
            best_rate = rate
            best_threads = t
    
    print()
    print("=" * 60)
    print(f"🏆 BEST: {best_threads} threads → {best_rate:,.0f} H/s")
    print()
    
    # Estimate mining time
    # Genesis target ≈ 2.18e63, meaning ~7.5e10 hashes per win
    est_hashes = 7.5e10
    est_hours = est_hashes / best_rate / 3600
    print(f"📊 Estimated time per mine (at genesis difficulty):")
    print(f"   ~{est_hashes:,.0f} hashes needed")
    print(f"   ~{est_hours:.1f} hours at {best_rate:,.0f} H/s")
    print()
    print(f"💡 Recommended config:")
    print(f"   \"threads\": {best_threads}")
    print(f"   \"batch_size\": {BATCH}")

else:
    # Python fallback benchmark
    print("\n⚠️  Running Python fallback (much slower)...")
    print("   Compile the C library for 10-30x speedup:")
    print("   gcc -O3 -march=native -mavx2 -funroll-loops -lpthread \\")
    print("       -o keccak_native.so -shared -fPIC keccak_native.c")
    print()
    
    try:
        from eth_hash.auto import keccak
        keccak_fn = keccak
        backend = "eth_hash"
    except ImportError:
        try:
            import hashlib
            keccak_fn = lambda d: hashlib.new('sha3_256', d).digest()
            backend = "hashlib (sha3_256 — NOT keccak!)"
        except Exception:
            print("❌ No keccak library available. Install: pip install eth-hash[pycryptodome]")
            sys.exit(1)
    
    BATCH_PY = 50_000
    print(f"[{backend}] Testing {BATCH_PY:,} hashes...")
    start = time.time()
    for n in range(BATCH_PY):
        nonce_bytes = n.to_bytes(32, 'big')
        keccak_fn(prefix + nonce_bytes)
    elapsed = time.time() - start
    rate = BATCH_PY / elapsed
    
    print(f"  → {elapsed:.3f}s | {rate:,.0f} H/s")
    print()
    
    est_hashes = 7.5e10
    est_hours = est_hashes / rate / 3600
    print(f"📊 At this speed, ~{est_hours:.0f} hours per mine")
    print(f"   (Compile C native for ~{est_hours / 15:.0f}x improvement)")

print()
print("Done!")
