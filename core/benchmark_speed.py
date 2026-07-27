import time
import numpy as np
from sentence_transformers import SentenceTransformer
import turbovec_lite_rs

# ---- Generate a larger dataset for a meaningful speed test ----

model = SentenceTransformer("all-MiniLM-L6-v2")
N = 5000
rng = np.random.default_rng(0)

# Reuse real embedding *distribution* by encoding random short phrases,
# then tiling/perturbing to reach N cheaply (speed test doesn't need semantic realism).
base_sentences = [f"sample text number {i}" for i in range(500)]
base_vectors = model.encode(base_sentences, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

reps = N // len(base_vectors) + 1
vectors = np.tile(base_vectors, (reps, 1))[:N]
vectors += rng.normal(scale=0.01, size=vectors.shape).astype(np.float32)  # jitter to avoid exact dupes
norms = np.linalg.norm(vectors, axis=1, keepdims=True)
vectors = (vectors / norms).astype(np.float32)

DIM = vectors.shape[1]
print(f"Dataset: {N} vectors, dim={DIM}")

# ---- Build Rust index ----
idx = turbovec_lite_rs.IdMapIndex(dim=DIM, seed=42)
idx.add_with_ids(vectors.flatten().tolist(), list(range(N)))

# ---- Benchmark: Rust module search ----
N_QUERIES = 100
query_indices = rng.choice(N, N_QUERIES, replace=False)
queries = [vectors[i].tolist() for i in query_indices]

start = time.perf_counter()
for q in queries:
    idx.search(q, 10)
rust_time = time.perf_counter() - start
print(f"\nRust module: {N_QUERIES} queries in {rust_time:.4f}s "
      f"({rust_time / N_QUERIES * 1000:.3f} ms/query)")

# ---- Benchmark: plain NumPy brute-force (uncompressed, exact) ----
start = time.perf_counter()
for i in query_indices:
    scores = vectors @ vectors[i]
    top10 = np.argsort(-scores)[:10]
rust_numpy_time = time.perf_counter() - start
print(f"NumPy brute-force: {N_QUERIES} queries in {rust_numpy_time:.4f}s "
      f"({rust_numpy_time / N_QUERIES * 1000:.3f} ms/query)")

print(f"\nSpeedup: {rust_numpy_time / rust_time:.2f}x")