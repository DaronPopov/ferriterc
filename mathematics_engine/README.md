# Ferrite Mathematics Engine

GPU-accelerated quantitative finance and large-scale computation control plane,
powered by the Ferrite OS TLSF allocator and wave-scheduled streaming.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    mathematics_engine/                            │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ monte_carlo/  │  │  portfolio/  │  │       risk/          │   │
│  │ path_pricer   │  │ covariance   │  │    var_engine        │   │
│  │              │  │   _stream    │  │                      │   │
│  │ 50M+ paths   │  │ 10K+ assets  │  │ 5M+ scenarios        │   │
│  │ streamed     │  │ tiled cov    │  │ streamed batches     │   │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘   │
│         │                 │                      │               │
│  ┌──────┴─────────────────┴──────────────────────┴───────────┐   │
│  │              TLSF Pool  ·  Wave Scheduling                │   │
│  │         O(1) alloc/free · bounded VRAM streaming          │   │
│  └──────┬─────────────────┬──────────────────────┬───────────┘   │
│         │                 │                      │               │
│  ┌──────┴───────┐  ┌──────┴───────┐  ┌──────────┴───────────┐   │
│  │    pde/      │  │   matrix/    │  │      greeks/         │   │
│  │ black_scholes│  │  streaming   │  │   adjoint_greeks     │   │
│  │    _fd       │  │ decomposition│  │                      │   │
│  │              │  │              │  │ 100K+ instruments    │   │
│  │ 1B+ grid pts │  │ 20K×20K+    │  │ bump-and-reval       │   │
│  │ slab-streamed│  │ tile-streamed│  │ batch-streamed       │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
│                                                                  │
│                    ┌───────────────────────┐                     │
│                    │    ferrite-os (GPU)    │                     │
│                    │  TLSF · streams · IPC │                     │
│                    └───────────────────────┘                     │
└──────────────────────────────────────────────────────────────────┘
```

## Modules

| Module | Script | Purpose |
|--------|--------|---------|
| `monte_carlo/` | `path_pricer.rs` | Stream 50M+ GBM paths for exotic option pricing (European, Asian, Barrier, Lookback) |
| `portfolio/` | `covariance_stream.rs` | Tile-streamed covariance matrix for 10K+ asset universes with portfolio optimization |
| `risk/` | `var_engine.rs` | VaR/CVaR via millions of historical/MC/stressed scenarios streamed through TLSF |
| `pde/` | `black_scholes_fd.rs` | Finite difference PDE solver on ultra-fine grids with slab-streamed spatial updates |
| `matrix/` | `streaming_decomposition.rs` | Block-Cholesky, LU, eigenvalue decomposition for matrices too large for VRAM |
| `greeks/` | `adjoint_greeks.rs` | Full Greeks surface (Δ,Γ,V,Θ,ρ,vanna,volga) for 100K+ instrument books |

## Usage

All scripts run via `ferrite-run`:

```bash
# Monte Carlo: 50M paths, Asian option
ferrite-run mathematics_engine/monte_carlo/path_pricer.rs \
  --paths 50000000 --steps 252 --option asian --spot 100.0 --strike 100.0

# Portfolio: 10K asset covariance matrix (400MB matrix, never in VRAM)
ferrite-run mathematics_engine/portfolio/covariance_stream.rs \
  --assets 10000 --observations 2520 --tile-size 512

# Risk: 5M scenario VaR
ferrite-run mathematics_engine/risk/var_engine.rs \
  --scenarios 5000000 --assets 2000 --confidence 0.99 --method montecarlo

# PDE: 50K×25K grid Black-Scholes (5GB grid, streamed)
ferrite-run mathematics_engine/pde/black_scholes_fd.rs \
  --spot-points 50000 --time-points 25000 --option call

# Matrix: 20K×20K Cholesky decomposition (1.6GB matrix, tiled)
ferrite-run mathematics_engine/matrix/streaming_decomposition.rs \
  --dim 20000 --tile-size 512 --method cholesky

# Greeks: 100K instrument book, full surface
ferrite-run mathematics_engine/greeks/adjoint_greeks.rs \
  --instruments 100000 --batch-size 10000
```

## Design Principles

1. **VRAM as throughput, not storage** — data streams through the GPU, never fully resident
2. **O(1) allocation** — TLSF gives deterministic alloc/free regardless of fragmentation
3. **Wave scheduling** — bounded concurrent shards across CUDA streams
4. **Compute > capacity** — problem sizes defined by compute budget, not memory capacity
5. **No Python in the critical path** — pure Rust from CLI to kernel dispatch
