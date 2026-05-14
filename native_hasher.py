"""
Native C Keccak256 Hasher — Python wrapper
Loads the compiled keccak_native.so for 10-30x speedup over Python.

Compile first:
  gcc -O3 -march=native -mavx2 -funroll-loops -lpthread -o keccak_native.so -shared -fPIC keccak_native.c
"""

import os
import ctypes

_native_lib = None


def _try_load_native():
    """Try loading our compiled C keccak library."""
    global _native_lib
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "keccak_native.so")
    if os.path.exists(lib_path):
        try:
            _native_lib = ctypes.CDLL(lib_path)
            _native_lib.mine_batch.restype = ctypes.c_int64
            _native_lib.mine_batch.argtypes = [
                ctypes.c_char_p,  # prefix (52 bytes: challenge + address)
                ctypes.c_char_p,  # target (32 bytes)
                ctypes.c_uint64,  # start_nonce
                ctypes.c_uint64,  # batch_size
            ]
            _native_lib.mine_batch_threaded.restype = ctypes.c_int64
            _native_lib.mine_batch_threaded.argtypes = [
                ctypes.c_char_p,  # prefix (52 bytes)
                ctypes.c_char_p,  # target (32 bytes)
                ctypes.c_uint64,  # start_nonce
                ctypes.c_uint64,  # batch_size
                ctypes.c_int,     # num_threads
            ]
            return True
        except Exception:
            pass
    return False


HAS_NATIVE = _try_load_native()


def mine_batch_native(prefix: bytes, target: bytes, start_nonce: int, batch_size: int, threads: int = 1) -> int:
    """
    Mine using C native extension. Returns nonce or -1 if not found.
    threads > 1 uses multi-threaded C mining (no Python GIL overhead).
    """
    if not HAS_NATIVE:
        return -1
    
    if threads > 1:
        result = _native_lib.mine_batch_threaded(
            prefix,
            target,
            ctypes.c_uint64(start_nonce),
            ctypes.c_uint64(batch_size),
            ctypes.c_int(threads)
        )
    else:
        result = _native_lib.mine_batch(
            prefix,
            target,
            ctypes.c_uint64(start_nonce),
            ctypes.c_uint64(batch_size)
        )
    return result


def get_hashrate_estimate():
    """Estimate hashrate based on available backend."""
    if HAS_NATIVE:
        cores = os.cpu_count() or 1
        # ~700K H/s per core at 2GHz, scales ~linearly
        est = cores * 700_000
        return f"~{est/1_000_000:.1f}M H/s (C native, {cores} cores)"
    try:
        import sha3
        return "~500K-2M H/s (pysha3)"
    except ImportError:
        return "~50K-100K H/s (eth_hash fallback — compile native for 10-30x boost!)"
