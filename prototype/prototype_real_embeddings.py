import itertools
import numpy as np
from sentence_transformers import SentenceTransformer

from prototype import (
    normalize,
    generate_random_rotation_matrix,
    apply_rotation,
    lloyd_max_quantizer_1d,
    brute_force_search,
    recall_at_k,
)

# ---- Generate unique sentences ----

subjects = ["The scientist", "The teacher", "The engineer", "The musician", "The athlete",
            "The chef", "The writer", "The doctor", "The pilot", "The farmer",
            "The student", "The artist", "The lawyer", "The nurse", "The architect",
            "The photographer", "The programmer", "The gardener", "The sailor", "The astronaut"]

actions = ["carefully examined the results", "explained the concept clearly",
           "worked late into the evening", "presented new findings today",
           "collaborated with the team", "solved a difficult problem",
           "traveled to a distant city", "wrote a detailed report",
           "discovered something unexpected", "prepared for the upcoming event",
           "reviewed the project plans", "shared insights with colleagues",
           "tested a new approach", "documented the entire process",
           "celebrated a recent success", "faced an unusual challenge",
           "learned a valuable lesson", "built something from scratch",
           "analyzed the available data", "reflected on past experiences"]

extra_actions = [a.replace("the", "a").replace("The", "A") for a in actions]

sentences = [f"{s} {a}." for s, a in itertools.product(subjects, actions)]
sentences += [f"{s} {a}." for s, a in itertools.product(subjects, extra_actions)][:600]

print(f"Unique sentences: {len(set(sentences))} / {len(sentences)}")

# ---- Encode ----

model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(sentences, convert_to_numpy=True).astype(np.float32)
print(f"Embeddings shape: {vectors.shape}")

# ---- Quantize + measure recall ----

DIM, N_VECTORS = vectors.shape[1], vectors.shape[0]
K, N_QUERIES = 10, 50

unit_vectors, _ = normalize(vectors)
rotation_matrix = generate_random_rotation_matrix(DIM)
rotated_vectors = apply_rotation(unit_vectors, rotation_matrix)
all_samples = rotated_vectors.flatten()

query_indices = np.random.default_rng(999).choice(N_VECTORS, N_QUERIES, replace=False)

print("\n--- Recall: 2-bit vs 4-bit ---")
for bits, n_levels in [(2, 4), (4, 16)]:
    boundaries, centroids = lloyd_max_quantizer_1d(all_samples, n_levels)
    bucket_idx = np.searchsorted(boundaries, rotated_vectors).astype(np.uint8)
    reconstructed = centroids[bucket_idx]

    recalls = [
        recall_at_k(
            brute_force_search(rotated_vectors[qi], rotated_vectors, K),
            brute_force_search(rotated_vectors[qi], reconstructed, K),
        )
        for qi in query_indices
    ]

    bytes_per_vec = DIM * bits / 8
    print(f"{bits}-bit: recall@{K}={np.mean(recalls):.4f}  "
          f"bytes/vec={bytes_per_vec:.0f}  compression={(DIM*4)/bytes_per_vec:.1f}x")