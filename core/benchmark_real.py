import itertools
import numpy as np
from sentence_transformers import SentenceTransformer
import turbovec_lite_rs

# ---- Generate unique sentences (same set as the Python prototype test) ----

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
vectors = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
print(f"Embeddings shape: {vectors.shape}")

N, DIM = vectors.shape
K, N_QUERIES = 10, 50

# ---- Build the Rust-backed index ----

idx = turbovec_lite_rs.IdMapIndex(dim=DIM, seed=42)
idx.add_with_ids(vectors.flatten().tolist(), list(range(N)))

# ---- Exact ground truth (brute-force cosine similarity in numpy) ----

def exact_top_k(query, database, k):
    scores = database @ query
    return set(np.argsort(-scores)[:k].tolist())

rng = np.random.default_rng(999)
query_indices = rng.choice(N, N_QUERIES, replace=False)

recalls = []
for qi in query_indices:
    ground_truth = exact_top_k(vectors[qi], vectors, K)
    _, approx_ids = idx.search(vectors[qi].tolist(), K)
    recalls.append(len(ground_truth & set(approx_ids)) / K)

print(f"\nRust/Python module — Recall@{K} over {N_QUERIES} queries: {np.mean(recalls):.4f}")