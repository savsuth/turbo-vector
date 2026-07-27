<p align="center">
  <img src="./docs/banner.png" alt="turbovec_lite_rs banner" width="100%" />
</p>

<p align="center">
  <a href="./LICENSE"><img alt="license" src="https://img.shields.io/badge/license-MIT-2ea043?style=flat-square"></a>
  <a href="https://www.rust-lang.org/"><img alt="rust" src="https://img.shields.io/badge/rust-000000?style=flat-square&logo=rust&logoColor=white"></a>
  <a href="https://www.python.org/"><img alt="python" src="https://img.shields.io/badge/python-3776AB?style=flat-square&logo=python&logoColor=white"></a>
</p>

---

**153.6 MB of float32 embeddings compresses to 12.2 MB — a 12.6x reduction, with no training step.**

### What this solves

Systems that search by meaning rather than exact keyword matching — semantic search, recommendation engines, chatbots that look things up before answering — represent every piece of text as a vector embedding: a list of a few hundred numbers that captures what that text means. Finding similar items means comparing a query's vector against every vector already stored. That comparison is cheap, but storing the vectors is not: each number is normally kept as a 32-bit float, so a single 384-number embedding takes about 1.5 KB, and a database of a few million documents can need tens of gigabytes of RAM just to hold the vectors, before any searching happens.

turbovec_lite_rs shrinks that footprint by compressing each number in a vector down to 2 bits instead of 32 — a technique called quantization — while keeping search results almost as accurate as uncompressed search would give. It uses TurboQuant, a quantization algorithm from Google Research, and needs no separate training or calibration step: vectors are compressed and made searchable the moment they are added. The core is written in Rust with NEON SIMD and multi-threaded search, and it is usable from both Rust and Python through PyO3 bindings.

Two folders, two stages of the same project:

- **`prototype/`** — a Python/NumPy implementation, used to work out and validate the algorithm before writing any Rust. Easier to debug, but not built for speed.
- **`core/`** — the Rust port of the same algorithm: SIMD, multi-threading, and Python bindings, built for speed.

- There is no training step — vectors are indexed as soon as they are added.
- The search kernel uses NEON SIMD, and its output is verified against a scalar fallback for correctness.
- Search runs in parallel across the index using `rayon`.
- Python bindings are provided through PyO3 and built with maturin.
- On 100K vectors, the fully serialized index achieves 12.6x compression.
- At 2-bit quantization, the index reaches 0.84 recall@10 — meaning it still finds 84% of the true top-10 nearest neighbors — against exact search on real sentence embeddings.

The example below shows the basic workflow: create an index, add vectors along with their ids, search it, remove an entry, and save it to disk so it can be reloaded later.

## Python

```bash
pip install maturin
maturin develop --release
```

```python
import turbovec_lite_rs

index = turbovec_lite_rs.IdMapIndex(dim=384, seed=42)
index.add_with_ids(vectors, ids)          # vectors: flat list, row-major (n * dim)
scores, result_ids = index.search(query, k=10)
index.remove(some_id)
index.write("index.tvim")
loaded = turbovec_lite_rs.IdMapIndex.load("index.tvim")
```

## Rust

```rust
use turbovec_lite_rs::IdMapIndex;

let mut index = IdMapIndex::new(384, 42);
index.add_with_ids(&vectors, &ids);
let (scores, result_ids) = index.search(&query, 10);
index.remove(some_id);
index.write("index.tvim").unwrap();
let loaded = IdMapIndex::load("index.tvim").unwrap();
```

## How it works

Compression works in five steps. The core idea is to reshape the vectors so that every number inside them follows a predictable statistical pattern — once that is true, they can be compressed aggressively without ever having to learn from the data first.

1. **Normalize.** Every vector is split into a direction and a magnitude — the direction is stored as a unit vector, and the magnitude is kept separately.
2. **Rotate.** All vectors are multiplied by the same fixed random orthogonal matrix. This rotation preserves the distances between vectors, but it also makes each coordinate's distribution statistically predictable, which is what the quantizer relies on.
3. **Quantize.** Because the post-rotation distribution is known in advance, the quantization buckets can be computed directly from the math rather than learned from data — 4 buckets for 2-bit precision, 16 for 4-bit.
4. **Pack.** Four 2-bit codes are packed into a single byte, giving roughly 16x raw compression before metadata is accounted for.
5. **Correct.** Quantization systematically shrinks the inner products between original and reconstructed vectors. To correct for this, a per-vector correction factor is computed once at insertion time, clamped to a safe range, and applied at search time.

At query time, the query vector is rotated into the same space and scored directly against the packed database — no decompression step is required.

## Results

<p align="center">
  <img src="./docs/compression_chart.png" width="70%" />
</p>

<p align="center">
  <img src="./docs/recall_chart.png" width="70%" />
</p>

The table below compares search speed across implementation stages, measured on 5K vectors at 384 dimensions over 100 queries on Apple Silicon, benchmarked against NumPy/BLAS:

| Version | Time/query | vs. NumPy |
|---|---|---|
| Scalar Rust, naive | 45.1 ms | 200x slower |
| Scalar Rust, fused | 25.2 ms | 108x slower |
| NEON, release | 0.588 ms | 2.7x slower |
| NEON + rayon, release | **0.461 ms** | **1.5x slower** |

## Bugs found along the way

- **The correction factor could go negative.** When a vector's reconstruction was nearly orthogonal to its original direction, the length-renormalization divisor was pushed toward zero, which in rare cases inverted the search ranking. This was fixed by clamping the correction factor to a bounded range.
- **Debug builds hid the SIMD gain entirely.** `maturin develop` compiles in debug mode by default, and under that mode the NEON kernel measured 92x slower than expected. Building with `--release` restored the intended performance.
- **A swapped tuple unpack caused a hard-to-trace panic.** Writing `DIM, N = vectors.shape` instead of `N, DIM = vectors.shape` on the Python side caused an out-of-bounds panic in Rust. Tracing it back to the single mismatched line in Python required following the full backtrace.

## Building

```bash
cd core
cargo test          # unit tests: rotation, quantization, packing, SIMD-vs-scalar parity
```

Running `cargo test` or `cargo build` directly will fail to link once the PyO3 bindings are part of the crate, since the `extension-module` feature intentionally skips linking against libpython. To verify the build, go through Python instead:

```bash
maturin develop --release
```

## Scope

This is not a production-ready library. Several things are intentionally out of scope for now: bit widths other than 2-bit, an x86 SIMD kernel (only NEON is implemented), filtered search, integrations with existing retrieval frameworks, and a formal benchmark against FAISS.

## References

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) — ICLR 2026
- [RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search](https://arxiv.org/abs/2405.12497) — SIGMOD 2024
