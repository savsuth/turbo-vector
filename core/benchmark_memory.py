import numpy as np
import turbovec_lite_rs

N = 100_000
DIM = 384

rng = np.random.default_rng(0)
vectors = rng.normal(size=(N, DIM)).astype(np.float32)
norms = np.linalg.norm(vectors, axis=1, keepdims=True)
vectors = (vectors / norms).astype(np.float32)

# Raw float32 size
raw_bytes = vectors.nbytes

# Build the compressed index
idx = turbovec_lite_rs.IdMapIndex(dim=DIM, seed=42)
idx.add_with_ids(vectors.flatten().tolist(), list(range(N)))
idx.write("mem_test.tvim")

import os
packed_bytes = os.path.getsize("mem_test.tvim")
os.remove("mem_test.tvim")

print(f"Vectors: {N:,}, dim: {DIM}")
print(f"Raw float32 size:     {raw_bytes:,} bytes  ({raw_bytes / 1e6:.1f} MB)")
print(f"Packed (2-bit) size:  {packed_bytes:,} bytes  ({packed_bytes / 1e6:.1f} MB)")
print(f"Compression ratio:    {raw_bytes / packed_bytes:.1f}x")