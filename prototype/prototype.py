import numpy as np

# ---- Stage 1: Generate test vectors and normalize ----

def generate_random_vectors(n_vectors: int, dim: int, seed: int = 42) -> np.ndarray:
    """Generate random vectors, roughly simulating embedding-like data."""
    rng = np.random.default_rng(seed)
    vectors = rng.normal(loc=0.0, scale=1.0, size=(n_vectors, dim)).astype(np.float32)
    return vectors


def normalize(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Strip the length (norm) from each vector, leaving unit direction vectors.
    Returns (unit_vectors, norms).
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit_vectors = vectors / norms
    return unit_vectors, norms.flatten()

def generate_random_rotation_matrix(dim: int, seed: int = 123) -> np.ndarray:
    """
    Generate a random orthogonal (rotation) matrix of shape (dim, dim).
    Uses QR decomposition of a random Gaussian matrix — a standard trick
    for sampling uniformly from the space of rotation matrices.
    """
    rng = np.random.default_rng(seed)
    random_matrix = rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(random_matrix)
    # Fix sign ambiguity so we get a proper rotation (det = +1), not a reflection
    d = np.sign(np.diag(r))
    q = q * d
    return q


def apply_rotation(vectors: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """Apply the same rotation matrix to every vector (row)."""
    return vectors @ rotation_matrix

def lloyd_max_quantizer_1d(samples: np.ndarray, n_levels: int, n_iters: int = 50):
    """
    Fit a 1D Lloyd-Max scalar quantizer to the given samples.
    Returns (boundaries, centroids).
    - centroids: the n_levels reconstruction values
    - boundaries: the n_levels - 1 decision boundaries between them
    """
    # Initialize centroids as evenly spaced points across the sample range
    centroids = np.linspace(samples.min(), samples.max(), n_levels)

    for _ in range(n_iters):
        # Step 1: boundaries are midpoints between adjacent centroids
        boundaries = (centroids[:-1] + centroids[1:]) / 2

        # Step 2: assign each sample to nearest centroid using boundaries
        bucket_indices = np.searchsorted(boundaries, samples)

        # Step 3: update each centroid to be the mean of samples assigned to it
        new_centroids = centroids.copy()
        for i in range(n_levels):
            bucket_samples = samples[bucket_indices == i]
            if len(bucket_samples) > 0:
                new_centroids[i] = bucket_samples.mean()

        if np.allclose(new_centroids, centroids, atol=1e-6):
            centroids = new_centroids
            break
        centroids = new_centroids

    boundaries = (centroids[:-1] + centroids[1:]) / 2
    return boundaries, centroids

def pack_2bit(bucket_indices: np.ndarray) -> np.ndarray:
    """
    Pack an array of 2-bit values (0-3) into bytes, 4 values per byte.
    Input shape: (n_vectors, dim) with values in [0, 3].
    Output shape: (n_vectors, ceil(dim / 4)) as uint8.
    """
    n_vectors, dim = bucket_indices.shape
    padded_dim = ((dim + 3) // 4) * 4
    padded = np.zeros((n_vectors, padded_dim), dtype=np.uint8)
    padded[:, :dim] = bucket_indices

    packed = np.zeros((n_vectors, padded_dim // 4), dtype=np.uint8)
    for i in range(4):
        packed |= (padded[:, i::4] << (2 * i)).astype(np.uint8)
    return packed


def unpack_2bit(packed: np.ndarray, dim: int) -> np.ndarray:
    """Reverse of pack_2bit — recover the original bucket indices."""
    n_vectors = packed.shape[0]
    padded_dim = packed.shape[1] * 4
    unpacked = np.zeros((n_vectors, padded_dim), dtype=np.uint8)
    for i in range(4):
        unpacked[:, i::4] = (packed >> (2 * i)) & 0b11
    return unpacked[:, :dim]


def quantize_1d(samples: np.ndarray, boundaries: np.ndarray, centroids: np.ndarray):
    """Assign each sample to a bucket index and its reconstruction value."""
    bucket_indices = np.searchsorted(boundaries, samples)
    reconstructed = centroids[bucket_indices]
    return bucket_indices, reconstructed

def brute_force_search(query: np.ndarray, database: np.ndarray, k: int) -> np.ndarray:
    """
    Return indices of the top-k vectors in `database` most similar to `query`,
    ranked by inner product (higher = more similar).
    """
    scores = database @ query
    top_k_indices = np.argsort(-scores)[:k]
    return top_k_indices


def recall_at_k(ground_truth: np.ndarray, approx: np.ndarray) -> float:
    """Fraction of ground_truth's top-k that appear in approx's top-k."""
    return len(set(ground_truth) & set(approx)) / len(ground_truth)


if __name__ == "__main__":
    N_VECTORS = 1000
    DIM = 128

    vectors = generate_random_vectors(N_VECTORS, DIM)
    print(f"Generated vectors shape: {vectors.shape}")
    print(f"First vector, first 5 dims: {vectors[0][:5]}")

    unit_vectors, norms = normalize(vectors)
    print(f"\nNorms shape: {norms.shape}")
    print(f"First 5 norms: {norms[:5]}")

    # Sanity check: unit vectors should have norm ~1.0
    check_norms = np.linalg.norm(unit_vectors, axis=1)
    print(f"\nSanity check — unit vector norms (should be ~1.0): {check_norms[:5]}")

    # ---- Stage 2: Random rotation ----
    rotation_matrix = generate_random_rotation_matrix(DIM)
    rotated_vectors = apply_rotation(unit_vectors, rotation_matrix)

    print(f"\nRotation matrix shape: {rotation_matrix.shape}")

    # Sanity check 1: rotation should preserve vector norms (still ~1.0)
    rotated_norms = np.linalg.norm(rotated_vectors, axis=1)
    print(f"Rotated vector norms (should still be ~1.0): {rotated_norms[:5]}")

    # Sanity check 2: rotation should preserve dot products (angles) between vectors
    original_dot = unit_vectors[0] @ unit_vectors[1]
    rotated_dot = rotated_vectors[0] @ rotated_vectors[1]
    print(f"\nDot product before rotation: {original_dot:.6f}")
    print(f"Dot product after rotation:  {rotated_dot:.6f}")
    print("(These should match closely — rotation preserves angles)")

    # Sanity check 3: individual coordinates should now look more Gaussian
    coord_before = unit_vectors[:, 0]
    coord_after = rotated_vectors[:, 0]
    print(f"\nCoordinate 0 std BEFORE rotation: {coord_before.std():.6f}")
    print(f"Coordinate 0 std AFTER rotation:  {coord_after.std():.6f}")
    print(f"(After rotation, std should be close to 1/sqrt(dim) = {1/np.sqrt(DIM):.6f})")

    # ---- Stage 3: Lloyd-Max quantization (2-bit = 4 levels) ----
    N_LEVELS = 4  # 2-bit

    # Fit the quantizer on coordinate 0 across all vectors
    coord0_samples = rotated_vectors[:, 0]
    boundaries, centroids = lloyd_max_quantizer_1d(coord0_samples, N_LEVELS)

    print(f"\n--- Stage 3: Lloyd-Max Quantization (2-bit, {N_LEVELS} levels) ---")
    print(f"Centroids: {centroids}")
    print(f"Boundaries: {boundaries}")

    bucket_indices, reconstructed = quantize_1d(coord0_samples, boundaries, centroids)
    print(f"\nFirst 10 original values:      {coord0_samples[:10]}")
    print(f"First 10 bucket indices:       {bucket_indices[:10]}")
    print(f"First 10 reconstructed values: {reconstructed[:10]}")

    mse = np.mean((coord0_samples - reconstructed) ** 2)
    print(f"\nMean squared error: {mse:.6f}")

    # ---- Stage 4: Quantize ALL dimensions with one shared quantizer, then bit-pack ----
    print(f"\n--- Stage 4: Full-vector quantization + bit-packing ---")

    # Fit one quantizer using samples pooled from every coordinate
    all_samples = rotated_vectors.flatten()
    boundaries_global, centroids_global = lloyd_max_quantizer_1d(all_samples, N_LEVELS)
    print(f"Global centroids: {centroids_global}")

    # Quantize every coordinate of every vector
    all_bucket_indices = np.searchsorted(boundaries_global, rotated_vectors).astype(np.uint8)
    print(f"Bucket indices shape: {all_bucket_indices.shape}")  # (1000, 128)

    # Bit-pack: 128 dims * 2 bits = 256 bits = 32 bytes per vector
    packed = pack_2bit(all_bucket_indices)
    print(f"Packed shape: {packed.shape}, dtype: {packed.dtype}")

    original_bytes = rotated_vectors.nbytes
    packed_bytes = packed.nbytes
    print(f"\nOriginal size (float32): {original_bytes:,} bytes")
    print(f"Packed size (2-bit):     {packed_bytes:,} bytes")
    print(f"Compression ratio:       {original_bytes / packed_bytes:.1f}x")

    # Verify unpacking recovers the same bucket indices
    unpacked = unpack_2bit(packed, DIM)
    assert np.array_equal(unpacked, all_bucket_indices), "Unpack mismatch!"
    print("\n✓ Unpacking correctly recovers original bucket indices")

    # Reconstruct actual vectors from quantized indices and measure overall error
    reconstructed_vectors = centroids_global[unpacked]
    overall_mse = np.mean((rotated_vectors - reconstructed_vectors) ** 2)
    print(f"\nOverall MSE across full vectors: {overall_mse:.6f}")

    # ---- Stage 5: Search quality — recall of quantized search vs. exact search ----
    print(f"\n--- Stage 5: Search recall (quantized vs. exact) ---")

    K = 10
    N_QUERIES = 50

    rng = np.random.default_rng(999)
    query_indices = rng.choice(N_VECTORS, size=N_QUERIES, replace=False)

    recalls = []
    for qi in query_indices:
        query = rotated_vectors[qi]  # use an existing rotated vector as a query

        # Ground truth: exact search over the original (uncompressed) rotated vectors
        ground_truth = brute_force_search(query, rotated_vectors, K)

        # Approximate: search using the RECONSTRUCTED (dequantized) vectors
        approx = brute_force_search(query, reconstructed_vectors, K)

        recalls.append(recall_at_k(ground_truth, approx))

    avg_recall = np.mean(recalls)
    print(f"Average Recall@{K} over {N_QUERIES} queries: {avg_recall:.4f}")
    print(f"(1.0 = perfect match with exact search, 0.0 = no overlap)")

    # ---- Stage 6: Compare 2-bit vs 4-bit ----
    print(f"\n--- Stage 6: 2-bit vs 4-bit comparison ---")

    for bits, n_levels in [(2, 4), (4, 16)]:
        boundaries_b, centroids_b = lloyd_max_quantizer_1d(all_samples, n_levels)
        bucket_idx_b = np.searchsorted(boundaries_b, rotated_vectors).astype(np.uint8)
        reconstructed_b = centroids_b[bucket_idx_b]

        recalls_b = []
        for qi in query_indices:
            query = rotated_vectors[qi]
            ground_truth = brute_force_search(query, rotated_vectors, K)
            approx = brute_force_search(query, reconstructed_b, K)
            recalls_b.append(recall_at_k(ground_truth, approx))

        avg_recall_b = np.mean(recalls_b)
        bytes_per_vector = (DIM * bits) / 8
        print(f"{bits}-bit: Recall@{K} = {avg_recall_b:.4f}, "
              f"bytes/vector = {bytes_per_vector:.0f}, "
              f"compression = {(DIM*4)/bytes_per_vector:.1f}x")