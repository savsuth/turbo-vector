import numpy as np
from sentence_transformers import SentenceTransformer
from turbovec_lite import IdMapIndex

sentences = [f"sentence number {i} about topic {i % 37}" for i in range(500)]
model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

idx = IdMapIndex(dim=vectors.shape[1], bit_width=2)
idx.add_with_ids(vectors, np.arange(len(vectors)))

scores, result_ids = idx.search(vectors[10], k=5)
print("Top-5 for query=vector[10]:", result_ids)
print("Id 10 in top-3:", 10 in result_ids[:3])

idx.write("test.tvim")
loaded = IdMapIndex.load("test.tvim")
_, ids2 = loaded.search(vectors[10], k=5)
print("Persistence matches:", np.array_equal(result_ids, ids2))

idx.remove(int(result_ids[0]))
print("After remove, count:", len(idx.ids))