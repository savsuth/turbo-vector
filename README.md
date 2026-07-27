<p align="center">
  <img src="./docs/banner.png" alt="turbovec_lite_rs banner" width="100%" />
</p>

# turbovec_lite_rs

A compressed vector search index, built to understand how Google Research's
**TurboQuant** algorithm works — a way to store embeddings in 2 bits per
dimension instead of 32, without training a model first.

## What I did

I implemented the algorithm in Python to make sure I understood it, then
ported it to Rust and sped it up with SIMD and multi-threading, then wrapped
the Rust core so it's callable from Python. Along the way I found and fixed
a real bug in the search-ranking math, and measured how well the compression
actually holds up on real data.

<p align="center">
  <img src="./docs/compression_chart.png" width="48%" />
  <img src="./docs/recall_chart.png" width="48%" />
</p>

## Process

1. **Python prototype** — validated normalize → rotate → quantize → search against exact brute-force search.
2. **Rust port** — same logic, rewritten as an `IdMapIndex` with add/search/remove/persistence, unit-tested.
3. **Python bindings** — via PyO3 + maturin, so the Rust core is callable from Python.
4. **SIMD + threading** — NEON kernel for the scoring loop, parallelized with `rayon`. Landed within 1.5x of NumPy's BLAS-backed search.

## Usage

```python
import turbovec_lite_rs

index = turbovec_lite_rs.IdMapIndex(dim=384, seed=42)
index.add_with_ids(vectors, ids)
scores, result_ids = index.search(query, k=10)
```

## Build

```bash
pip install maturin
maturin develop --release
```
