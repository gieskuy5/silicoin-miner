# ⛏️ Silicoin (SLC) CPU Miner

Optimized CPU miner for **Silicoin** — an ERC-20 PoW token mined by AI agents on Ethereum.

> Contract: [`0xbb572707D09eB2E80C835D3051097E5083D460Cc`](https://etherscan.io/address/0xbb572707D09eB2E80C835D3051097E5083D460Cc)
> Website: [silicoin.network](https://www.silicoin.network/)

## ⚡ Performance

Auto-detects CPU cores and scales accordingly:

| VPS Spec | Threads | Hashrate | Est. Time/Mine |
|----------|---------|----------|----------------|
| 1 core 1GHz (cheapest) | 1 | ~350K H/s | ~55h |
| 2 core 2GHz (basic) | 1-2 | ~700K-1.4M H/s | ~14-28h |
| 4 core 2.5GHz (mid) | 2-4 | ~2.5-5M H/s | ~4-8h |
| 8 core 3GHz (strong) | 4-8 | ~8-15M H/s | ~1.3-2.5h |
| 16 core 3.5GHz (beast) | 8-16 | ~20-40M H/s | ~30-60min |
| 64 core (dedicated) | 32-64 | ~80-150M H/s | ~8-15min |

*Estimates based on ~700K H/s per physical core at 2GHz. Actual results vary by CPU architecture.*

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/gieskuy5/silicoin-miner.git
cd silicoin-miner

# 2. Compile C native hasher (10-30x faster than Python)
gcc -O3 -march=native -mavx2 -funroll-loops -lpthread \
    -o keccak_native.so -shared -fPIC keccak_native.c

# 3. Install Python deps
pip install web3 eth-account eth-abi requests

# 4. Configure
cp config.json config.json.bak
# Edit config.json — set your private_key

# 5. Run
python3 miner.py
```

## ⚙️ Configuration

Edit `config.json`:

```json
{
  "rpc_url": "https://eth.llamarpc.com",
  "flashbots_rpc": "https://relay.flashbots.net",
  "contract_address": "0xbb572707D09eB2E80C835D3051097E5083D460Cc",
  "private_key": "0xYOUR_PRIVATE_KEY_HERE",
  "threads": 0,
  "batch_size": 0,
  "gas_limit_commit": 55000,
  "gas_limit_reveal": 120000,
  "max_priority_fee_gwei": 0.1,
  "max_fee_gwei": 5,
  "anchor_offset": 2,
  "auto_mine": true,
  "use_flashbots": true,
  "skip_if_gas_high": true,
  "max_gas_cost_usd": 0.50
}
```

### Config Options

| Key | Default | Description |
|-----|---------|-------------|
| `threads` | `0` (auto) | Mining threads. 0 = auto-detect from CPU cores |
| `batch_size` | `0` (auto) | Hashes per round. 0 = auto (2M × threads) |
| `use_flashbots` | `true` | Use Flashbots bundles (atomic commit+reveal) |
| `skip_if_gas_high` | `true` | Skip submission if gas exceeds limit |
| `max_gas_cost_usd` | `0.50` | Max gas cost per mine in USD |
| `max_fee_gwei` | `5` | Max gas price cap |
| `anchor_offset` | `2` | Blocks behind current for anchor hash |

### RPC Providers (free)

- `https://eth.llamarpc.com` (default, no key needed)
- `https://ethereum-rpc.publicnode.com`
- `https://rpc.ankr.com/eth`
- `https://1rpc.io/eth`

## 🔧 How It Works

```
┌─────────────────────────────────────────────────────┐
│  1. Fetch mineParams() from contract                │
│     → epochSeed, target, reward, epoch              │
│                                                     │
│  2. Compute challenge                               │
│     challenge = keccak256(anchorHash + epochSeed)    │
│                                                     │
│  3. Brute-force nonce (C native, multi-threaded)    │
│     Find: keccak256(challenge + addr + nonce) < target│
│                                                     │
│  4. Commit-Reveal (anti-frontrunning)               │
│     Block N:   commit(keccak256(nonce+secret+addr)) │
│     Block N+1: reveal(nonce, secret, anchorBlock)   │
│                                                     │
│  5. Collect reward: 1000 SLC 🎉                     │
└─────────────────────────────────────────────────────┘
```

### Why Commit-Reveal?

Prevents MEV bots from frontrunning your solution. Your nonce is hidden behind a commitment hash until the next block.

### Flashbots vs Sequential

- **Flashbots** (default): Atomic bundle — commit and reveal are guaranteed to land in consecutive blocks. No risk of reveal being sniped.
- **Sequential** (fallback): Send commit normally, wait 1 block, send reveal. Slightly riskier but works if Flashbots is down.

## 📁 Files

| File | Description |
|------|-------------|
| `miner.py` | Main miner — orchestration + blockchain interaction |
| `keccak_native.c` | C keccak256 kernel (multi-threaded, AVX2 optimized) |
| `native_hasher.py` | Python wrapper for C library |
| `config.json` | Configuration (edit this) |
| `benchmark.py` | Test your hashrate |

## 🧪 Benchmark Your Setup

```bash
python3 benchmark.py
```

This tests your hashrate without spending gas. Run it first to see what speed you'll get.

## 💰 Economics

| Metric | Value |
|--------|-------|
| Block reward | 1000 SLC (halving after 2500 mines) |
| Gas per mine | ~$0.05-0.50 (depends on network) |
| Target rate | ~15 mines/hour (network-wide) |
| Total supply | 10,000,000 SLC |
| Mineable | 5,000,000 SLC (50%) |

### Is It Profitable?

Depends on:
- Your hashrate vs network hashrate
- SLC price vs gas cost
- Current difficulty

Rule of thumb: if `(SLC_reward × SLC_price) > gas_cost`, you're profitable per mine.

## 🛡️ Security

- **Use a hot wallet** — never put your main wallet's private key here
- **Small ETH balance** — only keep enough for gas (~0.01-0.05 ETH)
- **config.json is gitignored** — your key won't be committed accidentally
- **Immutable contract** — no admin, no pause, no rug risk

## 📋 Requirements

- Linux x86_64 (AVX2 support recommended)
- GCC (for compiling C hasher)
- Python 3.8+
- Hot wallet with ~0.01 ETH for gas
- `web3`, `eth-account`, `eth-abi`, `requests`

### Without AVX2 (older CPUs)

```bash
# Compile without AVX2 flag
gcc -O3 -march=native -funroll-loops -lpthread \
    -o keccak_native.so -shared -fPIC keccak_native.c
```

### Without GCC (Python-only fallback)

The miner works without the C library — just much slower (~50K H/s vs ~700K+ H/s per core).

## 📜 License

MIT

## ⚠️ Disclaimer

This is experimental software for an experimental token. Mining involves real ETH gas costs. Only mine with funds you can afford to lose. DYOR.
