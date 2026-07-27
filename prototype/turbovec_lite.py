import numpy as np
import pickle


def _lloyd_max(samples: np.ndarray, n_levels: int, n_iters: int = 50):
    centroids = np.linspace(samples.min(), samples.max(), n_levels)
    for _ in range(n_iters):
        boundaries = (centroids[:-1] + centroids[1:]) / 2
        idx = np.searchsorted(boundaries, samples)
        new_centroids = np.array([
            samples[idx == i].mean() if np.any(idx == i) else centroids[i]
            for i in range(n_levels)
        ])
        if np.allclose(new_centroids, centroids, atol=1e-6):
            break
        centroids = new_centroids
    boundaries = (centroids[:-1] + centroids[1:]) / 2
    return boundaries, centroids


def _pack_2bit(bucket_idx: np.ndarray) -> np.ndarray:
    n, dim = bucket_idx.shape
    pad_dim = ((dim + 3) // 4) * 4
    padded = np.zeros((n, pad_dim), dtype=np.uint8)
    padded[:, :dim] = bucket_idx
    packed = np.zeros((n, pad_dim // 4), dtype=np.uint8)
    for i in range(4):
        packed |= (padded[:, i::4] << (2 * i)).astype(np.uint8)
    return packed


def _unpack_2bit(packed: np.ndarray, dim: int) -> np.ndarray:
    n = packed.shape[0]
    unpacked = np.zeros((n, packed.shape[1] * 4), dtype=np.uint8)
    for i in range(4):
        unpacked[:, i::4] = (packed >> (2 * i)) & 0b11
    return unpacked[:, :dim]


class IdMapIndex:
    """Compressed vector index with stable external ids, delete, and persistence."""

    def __init__(self, dim: int, bit_width: int = 2, seed: int = 42):
        assert bit_width == 2, "prototype supports 2-bit only"
        self.dim = dim
        self.bit_width = bit_width
        self.n_levels = 2 ** bit_width
        self.rng = np.random.default_rng(seed)
        self.rotation = self._make_rotation(dim, seed)

        self.ids: list[int] = []
        self.norms: list[float] = []
        self.packed: np.ndarray | None = None     # (n, dim*bits/8) uint8
        self.correction: list[float] = []          # length-renorm scalar per vector

        self.boundaries: np.ndarray | None = None
        self.centroids: np.ndarray | None = None
        self._fitted = False

    @staticmethod
    def _make_rotation(dim: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        q, r = np.linalg.qr(rng.normal(size=(dim, dim)))
        return q * np.sign(np.diag(r))

    def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray):
        vectors = vectors.astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1)
        unit = vectors / norms[:, None]
        rotated = unit @ self.rotation

        # Fit quantizer once, on first add — frozen after that (matches TQ+ design)
        if not self._fitted:
            self.boundaries, self.centroids = _lloyd_max(rotated.flatten(), self.n_levels)
            self._fitted = True

        bucket_idx = np.searchsorted(self.boundaries, rotated).astype(np.uint8)
        recon = self.centroids[bucket_idx]

        # Length-renormalization: correct systematic underestimation of inner products.
        # correction = ||v|| / <u, recon>, applied to scores at search time.
        dot_u_recon = np.sum(unit * recon, axis=1)
        dot_u_recon = np.where(np.abs(dot_u_recon) < 1e-8, 1.0, dot_u_recon)  # avoid div/0
        dot_u_recon = np.clip(dot_u_recon, 1e-3, None)
        correction = np.clip(norms / dot_u_recon, 0.5, 2.0)

        packed = _pack_2bit(bucket_idx)
        self.packed = packed if self.packed is None else np.vstack([self.packed, packed])
        self.ids.extend(int(i) for i in ids)
        self.norms.extend(norms.tolist())
        self.correction.extend(correction.tolist())

    def remove(self, id_: int):
        pos = self.ids.index(id_)
        self.ids.pop(pos)
        self.norms.pop(pos)
        self.correction.pop(pos)
        self.packed = np.delete(self.packed, pos, axis=0)

    def search(self, query: np.ndarray, k: int = 10):
        assert self._fitted, "index is empty"
        q_unit = query / np.linalg.norm(query)
        q_rotated = q_unit @ self.rotation

        bucket_idx = _unpack_2bit(self.packed, self.dim)
        recon = self.centroids[bucket_idx]                     # (n, dim)
        raw_scores = recon @ q_rotated                         # (n,)
        scores = raw_scores * np.array(self.correction)        # bias-corrected

        top = np.argsort(-scores)[:k]
        return scores[top], np.array(self.ids)[top]

    def write(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path: str) -> "IdMapIndex":
        obj = cls.__new__(cls)
        with open(path, "rb") as f:
            obj.__dict__.update(pickle.load(f))
        return obj