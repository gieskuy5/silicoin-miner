"""
GPU Miner for Silicoin — OpenCL backend
Supports NVIDIA (CUDA via OpenCL), AMD, Intel GPUs.

Requirements:
  pip install pyopencl numpy

For NVIDIA: install CUDA toolkit + OpenCL ICD
For AMD: install ROCm or AMDGPU-PRO drivers
For Intel: install Intel OpenCL runtime
"""

import os
import sys
import time
import secrets
import numpy as np

try:
    import pyopencl as cl
    HAS_OPENCL = True
except ImportError:
    HAS_OPENCL = False
    print("❌ pyopencl not installed. Run: pip install pyopencl")
    print("   Also need GPU drivers with OpenCL support.")
    sys.exit(1)


class GPUMiner:
    def __init__(self, device_idx=0, platform_idx=None, dry_run=False):
        """Initialize OpenCL context on specified GPU."""
        platforms = cl.get_platforms()
        
        if not platforms:
            raise RuntimeError("No OpenCL platforms found. Install GPU drivers.")
        
        # Auto-select platform with GPU device
        if platform_idx is not None:
            platform = platforms[platform_idx]
        else:
            platform = None
            for p in platforms:
                devices = p.get_devices(device_type=cl.device_type.GPU)
                if devices:
                    platform = p
                    break
            if platform is None:
                platform = platforms[0]
        
        devices = platform.get_devices(device_type=cl.device_type.GPU)
        if not devices:
            devices = platform.get_devices()
        
        if device_idx >= len(devices):
            device_idx = 0
        
        self.device = devices[device_idx]
        self.device_name = self.device.name.strip()
        self.compute_units = self.device.max_compute_units
        self.est_hashrate = self.compute_units * 64 * 800  # conservative
        
        # Global work size estimate
        self.global_work_size = min(
            2**22,
            self.device.max_work_group_size * self.compute_units * 64
        )
        
        if dry_run:
            return
        
        self.ctx = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.ctx)
        
        # Load and compile kernel
        kernel_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "keccak256.cl")
        with open(kernel_path) as f:
            kernel_src = f.read()
        
        self.program = cl.Program(self.ctx, kernel_src).build()
        
        self.max_work_group = self.device.max_work_group_size
        
        print(f"🖥️  GPU: {self.device_name}")
        print(f"   Platform: {platform.name}")
        print(f"   Compute units: {self.compute_units}")
        print(f"   Max work group: {self.max_work_group}")
        print(f"   Global work size: {self.global_work_size:,}")
    
    def mine_batch(self, prefix: bytes, target: bytes, start_nonce: int) -> int:
        """
        Run one batch on GPU. Returns found nonce or -1.
        
        prefix: 52 bytes (challenge + address)
        target: 32 bytes (big-endian)
        start_nonce: starting nonce for this batch
        """
        mf = cl.mem_flags
        
        # Create buffers
        prefix_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.frombuffer(prefix, dtype=np.uint8))
        target_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.frombuffer(target, dtype=np.uint8))
        
        # Result buffer: [found_flag, nonce_hi, nonce_lo]
        result_host = np.zeros(3, dtype=np.uint32)
        result_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=result_host)
        
        # Launch kernel
        self.program.mine_keccak256(
            self.queue,
            (self.global_work_size,),
            None,  # auto local work size
            prefix_buf,
            target_buf,
            np.uint64(start_nonce),
            result_buf
        )
        
        # Read result
        cl.enqueue_copy(self.queue, result_host, result_buf)
        self.queue.finish()
        
        if result_host[0]:
            nonce = (int(result_host[1]) << 32) | int(result_host[2])
            return nonce
        
        return -1
    
    def get_hashrate(self) -> float:
        """Benchmark: measure actual hashrate."""
        prefix = secrets.token_bytes(52)
        target = (2**200).to_bytes(32, 'big')  # very hard, won't find
        
        # Warmup
        self.mine_batch(prefix, target, 0)
        
        # Measure
        rounds = 5
        start = time.time()
        for i in range(rounds):
            self.mine_batch(prefix, target, i * self.global_work_size)
        elapsed = time.time() - start
        
        total_hashes = self.global_work_size * rounds
        return total_hashes / elapsed
    
    def benchmark(self, duration=3.0) -> float:
        """Run benchmark for `duration` seconds, return H/s."""
        prefix = secrets.token_bytes(52)
        target = (2**200).to_bytes(32, 'big')
        
        # Warmup
        self.mine_batch(prefix, target, 0)
        
        total_hashes = 0
        start = time.time()
        nonce = self.global_work_size
        
        while time.time() - start < duration:
            self.mine_batch(prefix, target, nonce)
            total_hashes += self.global_work_size
            nonce += self.global_work_size
        
        elapsed = time.time() - start
        return total_hashes / elapsed
    
    def info(self) -> dict:
        """Return GPU info dict."""
        return {
            "name": self.device.name,
            "compute_units": self.compute_units,
            "max_work_group": self.max_work_group,
            "global_work_size": self.global_work_size,
            "global_mem": self.device.global_mem_size // (1024**2),  # MB
        }


def list_devices():
    """List all available OpenCL devices."""
    platforms = cl.get_platforms()
    print(f"Found {len(platforms)} OpenCL platform(s):\n")
    
    for pi, platform in enumerate(platforms):
        print(f"  Platform {pi}: {platform.name}")
        devices = platform.get_devices()
        for di, device in enumerate(devices):
            dtype = "GPU" if device.type == cl.device_type.GPU else "CPU"
            mem = device.global_mem_size // (1024**2)
            print(f"    Device {di}: [{dtype}] {device.name} ({mem} MB, {device.max_compute_units} CU)")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("⛏️  SILICOIN GPU MINER — OpenCL")
    print("=" * 60)
    print()
    
    list_devices()
    
    try:
        gpu = GPUMiner()
        print()
        print("Running benchmark...")
        hashrate = gpu.get_hashrate()
        print(f"\n🏆 GPU Hashrate: {hashrate:,.0f} H/s ({hashrate/1_000_000:.1f}M H/s)")
        
        # Estimate
        est_hashes = 7.5e10
        est_time = est_hashes / hashrate
        print(f"⏱️  Estimated time per mine: {est_time/60:.1f} minutes")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Make sure GPU drivers and OpenCL runtime are installed.")
