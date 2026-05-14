"""
Silicoin (SLC) CPU/GPU Miner — Optimized with C Native Keccak256 + OpenCL GPU
Commit-Reveal scheme with Flashbots bundles for anti-frontrunning.

Auto-detects GPU (OpenCL) → falls back to CPU (C native) → Python fallback.
Supports: NVIDIA, AMD, Intel GPUs + any x86_64 CPU.
"""

import json
import os
import sys
import time
import secrets
import struct
import multiprocessing as mp
from multiprocessing import Process, Value, Array
from concurrent.futures import ProcessPoolExecutor
from web3 import Web3
from eth_account import Account
from eth_abi import encode as abi_encode
import requests

# ============================================================
# CONFIG
# ============================================================

def load_config():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    with open(config_path) as f:
        return json.load(f)

CONFIG = load_config()

# Auto-detect optimal settings based on CPU
CPU_COUNT = os.cpu_count() or 1

def get_optimal_threads():
    """Auto-detect optimal thread count. 0 in config = auto."""
    configured = CONFIG.get("threads", 0)
    if configured > 0:
        return configured
    # Use all physical cores (assume hyperthreading = 2x logical)
    # For mining, physical cores matter more than logical
    return max(1, CPU_COUNT // 2) if CPU_COUNT > 2 else CPU_COUNT

def get_optimal_batch_size():
    """Auto-detect batch size. 0 in config = auto."""
    configured = CONFIG.get("batch_size", 0)
    if configured > 0:
        return configured
    threads = get_optimal_threads()
    # Larger batches = less overhead, but more memory
    # ~2M per thread is sweet spot for L2 cache utilization
    return 2_000_000 * threads

# Contract ABI (minimal — only what we need)
MINE_ABI = json.loads('''[
    {"inputs":[],"name":"mineParams","outputs":[
        {"name":"epochSeed","type":"bytes32"},
        {"name":"target","type":"uint256"},
        {"name":"reward","type":"uint256"},
        {"name":"epoch","type":"uint8"},
        {"name":"poolLive","type":"bool"}
    ],"stateMutability":"view","type":"function"},
    {"inputs":[{"name":"commitment","type":"bytes32"}],"name":"commit","outputs":[],"stateMutability":"nonpayable","type":"function"},
    {"inputs":[{"name":"nonce","type":"uint256"},{"name":"secret","type":"bytes32"},{"name":"anchorBlock","type":"uint256"}],"name":"reveal","outputs":[],"stateMutability":"nonpayable","type":"function"},
    {"inputs":[],"name":"totalMined","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"currentReward","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"}
]''')


# ============================================================
# HASHING ENGINE (CPU optimized)
# ============================================================

def mine_worker(challenge_hex, miner_addr_hex, target_int, start_nonce, batch_size, result_nonce, found_flag):
    """
    Worker process: brute-force keccak256 in tight loop.
    Hash input: challenge(32) + address(20) + nonce(32) = 84 bytes
    """
    from eth_hash.auto import keccak
    
    challenge = bytes.fromhex(challenge_hex)
    miner_addr = bytes.fromhex(miner_addr_hex)
    target_bytes = target_int.to_bytes(32, 'big')
    
    nonce = start_nonce
    end_nonce = start_nonce + batch_size
    
    while nonce < end_nonce:
        if found_flag.value:
            return
        
        nonce_bytes = nonce.to_bytes(32, 'big')
        preimage = challenge + miner_addr + nonce_bytes
        h = keccak(preimage)
        
        if h < target_bytes:
            result_nonce.value = nonce
            found_flag.value = 1
            return
        
        nonce += 1


def mine_batch_optimized(challenge_hex, miner_addr_hex, target_int, start_nonce, batch_size):
    """
    Single-call optimized batch mining.
    Priority: GPU (OpenCL) > C native multi-threaded > pysha3 > eth_hash
    """
    challenge = bytes.fromhex(challenge_hex)
    miner_addr = bytes.fromhex(miner_addr_hex)
    target_bytes = target_int.to_bytes(32, 'big')
    prefix = challenge + miner_addr  # 52 bytes
    
    # Try GPU first (fastest — 100-500x over CPU)
    if not CONFIG.get("force_cpu", False):
        try:
            from gpu.gpu_miner import GPUMiner
            global _gpu_instance
            if '_gpu_instance' not in globals() or _gpu_instance is None:
                _gpu_instance = GPUMiner(device_idx=CONFIG.get("gpu_device", 0))
            result = _gpu_instance.mine_batch(prefix, target_bytes, start_nonce)
            return result if result >= 0 else None
        except (ImportError, RuntimeError, Exception):
            pass  # No GPU available, fall through
    
    # Try C native (10-30x over Python)
    try:
        from native_hasher import mine_batch_native, HAS_NATIVE
    except ImportError:
        HAS_NATIVE = False
    
    if HAS_NATIVE:
        threads = get_optimal_threads()
        result = mine_batch_native(prefix, target_bytes, start_nonce, batch_size, threads=threads)
        return result if result >= 0 else None
    
    # Fallback: Python keccak
    try:
        import sha3
        keccak_fn = lambda data: sha3.keccak_256(data).digest()
    except ImportError:
        from eth_hash.auto import keccak
        keccak_fn = keccak
    
    nonce = start_nonce
    end_nonce = start_nonce + batch_size
    
    while nonce < end_nonce:
        nonce_bytes = nonce.to_bytes(32, 'big')
        h = keccak_fn(prefix + nonce_bytes)
        if h < target_bytes:
            return nonce
        nonce += 1
    
    return None


# ============================================================
# BLOCKCHAIN INTERACTION
# ============================================================

class SilicoinMiner:
    def __init__(self):
        self.w3 = Web3(Web3.HTTPProvider(CONFIG["rpc_url"]))
        self.contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(CONFIG["contract_address"]),
            abi=MINE_ABI
        )
        self.account = Account.from_key(CONFIG["private_key"])
        self.miner_addr = self.account.address
        self.threads = get_optimal_threads()
        self.batch_size = get_optimal_batch_size()
        
        # Stats
        self.total_hashes = 0
        self.start_time = time.time()
        self.mines_won = 0
        
    def log(self, msg):
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)
    
    def _get_eth_price(self):
        """Get ETH price in USD (cached 5 min)."""
        now = time.time()
        if hasattr(self, '_eth_price_cache') and now - self._eth_price_ts < 300:
            return self._eth_price_cache
        try:
            r = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd", timeout=5)
            price = r.json()["ethereum"]["usd"]
            self._eth_price_cache = price
            self._eth_price_ts = now
            return price
        except Exception:
            return getattr(self, '_eth_price_cache', 2500)  # fallback
    
    def get_mine_params(self):
        """Fetch current mining parameters from contract."""
        result = self.contract.functions.mineParams().call()
        return {
            "epoch_seed": result[0],
            "target": result[1],
            "reward": result[2],
            "epoch": result[3],
            "pool_live": result[4]
        }
    
    def compute_challenge(self, anchor_block):
        """challenge = keccak256(anchorHash ++ epochSeed)"""
        anchor_hash = self.w3.eth.get_block(anchor_block)["hash"]
        params = self.get_mine_params()
        
        if not params["pool_live"]:
            self.log("⚠️  Pool not live yet, waiting...")
            return None, None
        
        challenge = Web3.solidity_keccak(
            ["bytes32", "bytes32"],
            [anchor_hash, params["epoch_seed"]]
        )
        
        return challenge, params
    
    def find_nonce_multiprocess(self, challenge, target):
        """Multi-process nonce search (Python-level parallelism, for non-native fallback)."""
        challenge_hex = challenge.hex()
        miner_hex = self.miner_addr[2:].lower()
        
        found_flag = Value('i', 0)
        result_nonce = Value('l', 0)
        
        batch_per_worker = self.batch_size // self.threads
        round_num = 0
        
        while not found_flag.value:
            processes = []
            for i in range(self.threads):
                start = (round_num * self.threads + i) * batch_per_worker + secrets.randbelow(2**32)
                p = Process(
                    target=mine_worker,
                    args=(challenge_hex, miner_hex, target, start, batch_per_worker, result_nonce, found_flag)
                )
                processes.append(p)
                p.start()
            
            for p in processes:
                p.join()
            
            self.total_hashes += self.threads * batch_per_worker
            round_num += 1
            
            elapsed = time.time() - self.start_time
            hashrate = self.total_hashes / elapsed if elapsed > 0 else 0
            self.log(f"⛏️  Hashing... {self.total_hashes:,} hashes | {hashrate:,.0f} H/s | Round {round_num}")
        
        return result_nonce.value
    
    def find_nonce_single(self, challenge, target):
        """Single-call optimized mining (C native handles threading internally)."""
        challenge_hex = challenge.hex()
        miner_hex = self.miner_addr[2:].lower()
        
        round_num = 0
        batch = self.batch_size
        
        while True:
            start_nonce = round_num * batch + secrets.randbelow(2**32)
            result = mine_batch_optimized(challenge_hex, miner_hex, target, start_nonce, batch)
            
            self.total_hashes += batch
            round_num += 1
            
            if result is not None:
                return result
            
            # Print stats every 5 rounds
            elapsed = time.time() - self.start_time
            hashrate = self.total_hashes / elapsed if elapsed > 0 else 0
            if round_num % 5 == 0:
                self.log(f"⛏️  {self.total_hashes:,} hashes | {hashrate:,.0f} H/s | Round {round_num}")
    
    def build_commit_reveal_bundle(self, nonce, anchor_block):
        """Build Flashbots bundle: commit in block N, reveal in block N+1."""
        secret = secrets.token_bytes(32)
        
        # Commitment = keccak256(nonce ++ secret ++ minerAddr ++ anchorBlock)
        commitment = Web3.solidity_keccak(
            ["uint256", "bytes32", "address", "uint256"],
            [nonce, secret, self.miner_addr, anchor_block]
        )
        
        # Get current tx nonce
        tx_nonce = self.w3.eth.get_transaction_count(self.miner_addr, "pending")
        
        # Gas params
        base_fee = self.w3.eth.get_block("latest")["baseFeePerGas"]
        max_priority = Web3.to_wei(CONFIG["max_priority_fee_gwei"], "gwei")
        max_fee = Web3.to_wei(CONFIG["max_fee_gwei"], "gwei")
        
        # Check gas cost before submitting
        if CONFIG.get("skip_if_gas_high", False):
            total_gas = CONFIG["gas_limit_commit"] + CONFIG["gas_limit_reveal"]
            estimated_cost_eth = (base_fee + max_priority) * total_gas / 1e18
            eth_price = self._get_eth_price()
            estimated_cost_usd = estimated_cost_eth * eth_price
            max_cost = CONFIG.get("max_gas_cost_usd", 0.50)
            
            if estimated_cost_usd > max_cost:
                self.log(f"⚠️  Gas too high: ${estimated_cost_usd:.3f} > ${max_cost:.2f} limit. Waiting...")
                return None, None, None, None
        
        # TX 1: commit
        commit_tx = self.contract.functions.commit(commitment).build_transaction({
            "from": self.miner_addr,
            "nonce": tx_nonce,
            "gas": CONFIG["gas_limit_commit"],
            "maxFeePerGas": max_fee,
            "maxPriorityFeePerGas": max_priority,
            "chainId": 1,
            "type": 2
        })
        signed_commit = self.account.sign_transaction(commit_tx)
        
        # TX 2: reveal (tx nonce + 1)
        reveal_tx = self.contract.functions.reveal(nonce, secret, anchor_block).build_transaction({
            "from": self.miner_addr,
            "nonce": tx_nonce + 1,
            "gas": CONFIG["gas_limit_reveal"],
            "maxFeePerGas": max_fee,
            "maxPriorityFeePerGas": max_priority,
            "chainId": 1,
            "type": 2
        })
        signed_reveal = self.account.sign_transaction(reveal_tx)
        
        return signed_commit, signed_reveal, secret, commitment
    
    def send_flashbots_bundle(self, signed_commit, signed_reveal, target_block):
        """Send bundle via Flashbots relay."""
        # Flashbots bundle for block N (commit)
        payload_commit = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "eth_sendBundle",
            "params": [{
                "txs": ["0x" + signed_commit.raw_transaction.hex()],
                "blockNumber": hex(target_block),
            }]
        }
        
        # Bundle for block N+1 (reveal)
        payload_reveal = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "eth_sendBundle",
            "params": [{
                "txs": ["0x" + signed_reveal.raw_transaction.hex()],
                "blockNumber": hex(target_block + 1),
            }]
        }
        
        # Sign with ephemeral Flashbots auth key
        fb_key = Account.create()
        headers = {
            "Content-Type": "application/json",
            "X-Flashbots-Signature": f"{fb_key.address}:{fb_key.key.hex()}"
        }
        
        try:
            r1 = requests.post(CONFIG["flashbots_rpc"], json=payload_commit, headers=headers, timeout=10)
            r2 = requests.post(CONFIG["flashbots_rpc"], json=payload_reveal, headers=headers, timeout=10)
            self.log(f"📦 Bundle sent: commit→block {target_block}, reveal→block {target_block+1}")
            self.log(f"   Commit response: {r1.status_code}")
            self.log(f"   Reveal response: {r2.status_code}")
            return True
        except Exception as e:
            self.log(f"❌ Flashbots error: {e}")
            return False
    
    def send_sequential(self, signed_commit, signed_reveal):
        """Fallback: send commit, wait 1 block, send reveal."""
        try:
            # Send commit
            commit_hash = self.w3.eth.send_raw_transaction(signed_commit.raw_transaction)
            self.log(f"📤 Commit sent: {commit_hash.hex()}")
            
            # Wait for commit to be mined
            receipt = self.w3.eth.wait_for_transaction_receipt(commit_hash, timeout=60)
            if receipt["status"] != 1:
                self.log("❌ Commit failed!")
                return False
            
            commit_block = receipt["blockNumber"]
            self.log(f"✅ Commit mined in block {commit_block}")
            
            # Wait for next block
            self.log("⏳ Waiting for next block for reveal...")
            while self.w3.eth.block_number <= commit_block:
                time.sleep(1)
            
            # Send reveal
            reveal_hash = self.w3.eth.send_raw_transaction(signed_reveal.raw_transaction)
            self.log(f"📤 Reveal sent: {reveal_hash.hex()}")
            
            receipt = self.w3.eth.wait_for_transaction_receipt(reveal_hash, timeout=60)
            if receipt["status"] == 1:
                self.log(f"🎉 MINED SUCCESSFULLY! Reveal in block {receipt['blockNumber']}")
                self.mines_won += 1
                return True
            else:
                self.log("❌ Reveal failed (window missed or target changed)")
                return False
                
        except Exception as e:
            self.log(f"❌ TX error: {e}")
            return False
    
    def run(self):
        """Main mining loop."""
        self.log("=" * 60)
        self.log("⛏️  SILICOIN MINER — CPU/GPU")
        self.log(f"   Miner: {self.miner_addr}")
        self.log(f"   CPU cores detected: {CPU_COUNT}")
        self.log(f"   Mining threads: {self.threads}")
        self.log(f"   Batch size: {self.batch_size:,}")
        self.log(f"   Contract: {CONFIG['contract_address']}")
        
        # Detect mining backend
        self.mining_mode = "python"
        
        if not CONFIG.get("force_cpu", False):
            try:
                from gpu.gpu_miner import GPUMiner
                gpu = GPUMiner(device_idx=CONFIG.get("gpu_device", 0))
                self.log(f"   🖥️  GPU: ✅ {gpu.device.name} ({gpu.compute_units} CU)")
                self.log(f"   Mode: GPU (OpenCL) — ~{gpu.global_work_size:,} hashes/batch")
                self.mining_mode = "gpu"
            except Exception:
                pass
        
        if self.mining_mode != "gpu":
            try:
                from native_hasher import HAS_NATIVE, get_hashrate_estimate
                if HAS_NATIVE:
                    self.log(f"   🔧 Backend: C native ({self.threads} threads)")
                    self.log(f"   Estimated speed: {get_hashrate_estimate()}")
                    self.mining_mode = "cpu_native"
                else:
                    self.log("   ⚠️  C native not compiled — using Python fallback (slow!)")
                    self.log("   Compile: gcc -O3 -march=native -mavx2 -funroll-loops -lpthread -o keccak_native.so -shared -fPIC keccak_native.c")
            except ImportError:
                self.log("   ⚠️  native_hasher.py not found — using Python fallback (slow!)")
        
        self.log("=" * 60)
        
        while True:
            try:
                # 1. Get current block and params
                current_block = self.w3.eth.block_number
                anchor_block = current_block - CONFIG.get("anchor_offset", 2)
                
                challenge, params = self.compute_challenge(anchor_block)
                if challenge is None:
                    time.sleep(12)
                    continue
                
                target = params["target"]
                reward = params["reward"] / 10**18
                
                self.log(f"🎯 Target: {target:.2e} | Reward: {reward:.0f} SLC | Epoch: {params['epoch']}")
                self.log(f"📦 Anchor block: {anchor_block}")
                
                # Estimate time to find nonce
                estimated_hashes = (2**256) // target if target > 0 else 0
                est_time = estimated_hashes / (self.batch_size * 0.9) if self.batch_size > 0 else 0
                if est_time > 0:
                    self.log(f"⏱️  Estimated: ~{estimated_hashes:,.0f} hashes (~{est_time/3600:.1f}h at current speed)")
                
                # 2. Find valid nonce
                self.start_time = time.time()
                self.total_hashes = 0
                
                # Use single-call method (C native handles threading internally)
                try:
                    from native_hasher import HAS_NATIVE
                    use_native = HAS_NATIVE
                except ImportError:
                    use_native = False
                
                if use_native:
                    nonce = self.find_nonce_single(challenge, target)
                elif self.threads > 1:
                    nonce = self.find_nonce_multiprocess(challenge, target)
                else:
                    nonce = self.find_nonce_single(challenge, target)
                
                elapsed = time.time() - self.start_time
                self.log(f"✅ Nonce found: {nonce} in {elapsed:.1f}s ({self.total_hashes:,} hashes)")
                
                # 3. Re-verify params haven't changed
                new_params = self.get_mine_params()
                if new_params["target"] != target or new_params["epoch"] != params["epoch"]:
                    self.log("⚠️  Params changed during mining, discarding nonce...")
                    continue
                
                # 4. Build and send commit-reveal
                result = self.build_commit_reveal_bundle(nonce, anchor_block)
                if result[0] is None:
                    # Gas too high, retry later
                    time.sleep(30)
                    continue
                
                signed_commit, signed_reveal, secret, commitment = result
                
                # Try Flashbots first, fallback to sequential
                target_block = self.w3.eth.block_number + 1
                
                if CONFIG.get("use_flashbots", True):
                    success = self.send_flashbots_bundle(signed_commit, signed_reveal, target_block)
                    if not success:
                        self.log("⚠️  Flashbots failed, trying sequential...")
                        success = self.send_sequential(signed_commit, signed_reveal)
                else:
                    success = self.send_sequential(signed_commit, signed_reveal)
                
                if success:
                    self.log(f"🏆 Total mines won: {self.mines_won}")
                
                # Brief cooldown
                time.sleep(2)
                
            except KeyboardInterrupt:
                self.log("🛑 Miner stopped by user")
                break
            except Exception as e:
                self.log(f"❌ Error: {e}")
                time.sleep(5)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="⛏️ Silicoin (SLC) Miner — CPU/GPU")
    parser.add_argument("--gpu", action="store_true", help="Force GPU mining (OpenCL)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mining (skip GPU detection)")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark only, don't mine")
    parser.add_argument("--threads", type=int, default=0, help="Override thread count (0=auto)")
    parser.add_argument("--device", type=int, default=0, help="GPU device index (default: 0)")
    args = parser.parse_args()
    
    # Override config with CLI args
    if args.threads > 0:
        CONFIG["threads"] = args.threads
    if args.cpu:
        CONFIG["force_cpu"] = True
    if args.gpu:
        CONFIG["force_gpu"] = True
    CONFIG["gpu_device"] = args.device
    
    if args.benchmark:
        print("=" * 60)
        print("⛏️  SILICOIN MINER — BENCHMARK MODE")
        print("=" * 60)
        print()
        
        # Try GPU
        if not args.cpu:
            try:
                from gpu.gpu_miner import GPUMiner
                gpu = GPUMiner(device_idx=args.device)
                print("\nBenchmarking GPU...")
                rate = gpu.get_hashrate()
                print(f"🏆 GPU: {rate:,.0f} H/s ({rate/1_000_000:.1f}M H/s)")
                est = 7.5e10 / rate / 60
                print(f"⏱️  Est. time per mine: ~{est:.1f} minutes")
            except Exception as e:
                print(f"GPU: ❌ {e}")
        
        # CPU benchmark
        if not args.gpu:
            print("\nBenchmarking CPU...")
            try:
                from native_hasher import mine_batch_native, HAS_NATIVE
                if HAS_NATIVE:
                    import secrets as _s
                    prefix = _s.token_bytes(52)
                    target = (2**200).to_bytes(32, 'big')
                    BATCH = 2_000_000
                    threads = get_optimal_threads()
                    
                    start = time.time()
                    mine_batch_native(prefix, target, 0, BATCH, threads=threads)
                    elapsed = time.time() - start
                    rate = BATCH / elapsed
                    print(f"🏆 CPU ({threads} threads, C native): {rate:,.0f} H/s ({rate/1_000_000:.2f}M H/s)")
                    est = 7.5e10 / rate / 3600
                    print(f"⏱️  Est. time per mine: ~{est:.1f} hours")
                else:
                    print("C native not compiled. Run: gcc -O3 -march=native -mavx2 -funroll-loops -lpthread -o keccak_native.so -shared -fPIC keccak_native.c")
            except ImportError:
                print("native_hasher.py not found")
        
        sys.exit(0)
    
    if not CONFIG.get("private_key") or CONFIG["private_key"] == "0xYOUR_PRIVATE_KEY_HERE":
        print("❌ Set 'private_key' in config.json first!")
        print("   Use a hot wallet with small ETH balance for gas.")
        print(f"   Detected {CPU_COUNT} CPU cores → will use {get_optimal_threads()} mining threads")
        print()
        print("Usage:")
        print("  python3 miner.py              # Auto-detect GPU/CPU")
        print("  python3 miner.py --gpu        # Force GPU mode")
        print("  python3 miner.py --cpu        # Force CPU mode")
        print("  python3 miner.py --benchmark  # Test hashrate only")
        sys.exit(1)
    
    miner = SilicoinMiner()
    miner.run()
