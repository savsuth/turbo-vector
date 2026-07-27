# turbovec_lite_rs

`turbovec_lite_rs` is a lightweight Rust vector search library designed for fast approximate nearest-neighbor retrieval using 2-bit quantization, scalar and NEON scoring, and optional Python bindings.

## Features

- `IdMapIndex` for storing vectors with integer ids
- 2-bit quantization with per-vector correction
- Rotation-based encoding for compact scoring
- Multi-threaded search via `rayon`
- Python extension module via `pyo3`
- Serialization and loading of trained indexes

## Repository Structure

- `Cargo.toml` — Rust package manifest
- `src/lib.rs` — library implementation and Python bindings
- `benchmark_real.py` — example benchmark using SentenceTransformers and the Python extension module
- `benchmark_speed.py`, `benchmark_memory.py` — benchmark scripts (if present)

## Build

### Rust

```bash
cargo build
```

### Release

```bash
cargo build --release
```

## Python bindings

This crate builds a Python extension module named `turbovec_lite_rs` using `pyo3`.

### Build Python package locally

```bash
maturin develop --release
```

### Use from Python

```python
import turbovec_lite_rs

idx = turbovec_lite_rs.IdMapIndex(dim=384, seed=42)
idx.add_with_ids(vectors, ids)
scores, result_ids = idx.search(query_vector, k=10)
```

## Example usage

From Rust:

```rust
use turbovec_lite_rs::IdMapIndex;

let mut idx = IdMapIndex::new(384, 42);
idx.add_with_ids(&vectors, &ids);
let (scores, ids) = idx.search(&query, 10);
```

From Python:

```python
import turbovec_lite_rs

idx = turbovec_lite_rs.IdMapIndex(dim=384, seed=42)
idx.add_with_ids(vectors, ids)
(scores, ids) = idx.search(query, 10)
```

## API

### `IdMapIndex`

- `new(dim: usize, seed: u64)` — create a new index
- `add_with_ids(vectors: &[f32], ids: &[u64])` — add vectors and matching ids
- `search(query: &[f32], k: usize) -> (Vec<f32>, Vec<u64>)` — search for top-`k` matches
- `remove(id: u64)` — remove a vector by id
- `write(path: &str)` — serialize the index
- `load(path: &str) -> IdMapIndex` — deserialize an index from disk

## Benchmarks

The repository includes Python benchmark scripts such as `benchmark_real.py`.

`benchmark_real.py` builds a Python dataset with 600 sentence embeddings and compares the Rust-backed index performance and recall against brute-force cosine similarity.

## Testing

Run Rust tests with:

```bash
cargo test
```

## Requirements

- Rust toolchain (`cargo`)
- Python for Python binding builds and benchmarks
- `maturin` for building the Python extension

## Notes

- The implementation uses a rotation matrix and 4-level quantization.
- CPU-specific NEON scoring is enabled on `aarch64` targets, while scalar fallback is used elsewhere.
- The Python wrapper exposes the same `IdMapIndex` interface as the Rust library.
