use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};

pub fn normalize(v: &[f32]) -> (Vec<f32>, f32) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    (v.iter().map(|x| x / norm).collect(), norm)
}

pub fn generate_rotation_matrix(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut a = vec![0.0f32; dim * dim];
    for x in a.iter_mut() {
        *x = StandardNormal.sample(&mut rng);
    }
    let mut q = vec![0.0f32; dim * dim];
    for col in 0..dim {
        let mut v: Vec<f32> = (0..dim).map(|row| a[row * dim + col]).collect();
        for prev in 0..col {
            let q_prev: Vec<f32> = (0..dim).map(|row| q[row * dim + prev]).collect();
            let dot: f32 = v.iter().zip(&q_prev).map(|(x, y)| x * y).sum();
            for row in 0..dim {
                v[row] -= dot * q_prev[row];
            }
        }
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for row in 0..dim {
            q[row * dim + col] = v[row] / norm;
        }
    }
    q
}

pub fn apply_rotation(v: &[f32], rotation: &[f32], dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; dim];
    for row in 0..dim {
        let mut sum = 0.0f32;
        for col in 0..dim {
            sum += v[col] * rotation[row * dim + col];
        }
        out[row] = sum;
    }
    out
}

pub fn lloyd_max_quantizer(samples: &[f32], n_levels: usize, n_iters: usize) -> (Vec<f32>, Vec<f32>) {
    let min = samples.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = samples.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut centroids: Vec<f32> = (0..n_levels)
        .map(|i| min + (max - min) * (i as f32) / (n_levels as f32 - 1.0))
        .collect();

    for _ in 0..n_iters {
        let boundaries: Vec<f32> = (0..n_levels - 1)
            .map(|i| (centroids[i] + centroids[i + 1]) / 2.0)
            .collect();
        let mut sums = vec![0.0f32; n_levels];
        let mut counts = vec![0usize; n_levels];
        for &s in samples {
            let bucket = boundaries.partition_point(|&b| b <= s);
            sums[bucket] += s;
            counts[bucket] += 1;
        }
        let mut changed = false;
        for i in 0..n_levels {
            if counts[i] > 0 {
                let new_val = sums[i] / counts[i] as f32;
                if (new_val - centroids[i]).abs() > 1e-6 {
                    changed = true;
                }
                centroids[i] = new_val;
            }
        }
        if !changed {
            break;
        }
    }
    let boundaries: Vec<f32> = (0..n_levels - 1)
        .map(|i| (centroids[i] + centroids[i + 1]) / 2.0)
        .collect();
    (boundaries, centroids)
}

pub fn quantize_value(value: f32, boundaries: &[f32]) -> u8 {
    boundaries.partition_point(|&b| b <= value) as u8
}

pub fn pack_2bit(bucket_indices: &[u8]) -> Vec<u8> {
    let packed_len = (bucket_indices.len() + 3) / 4;
    let mut packed = vec![0u8; packed_len];
    for (i, &b) in bucket_indices.iter().enumerate() {
        packed[i / 4] |= b << ((i % 4) * 2);
    }
    packed
}

pub fn unpack_2bit(packed: &[u8], dim: usize) -> Vec<u8> {
    (0..dim).map(|i| (packed[i / 4] >> ((i % 4) * 2)) & 0b11).collect()
}

/// Scalar fallback: unpack 2-bit codes and score against query in one pass.
fn score_scalar(packed: &[u8], centroids: &[f32], query: &[f32], dim: usize) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..dim {
        let bucket = (packed[i / 4] >> ((i % 4) * 2)) & 0b11;
        sum += centroids[bucket as usize] * query[i];
    }
    sum
}

#[cfg(target_arch = "aarch64")]
mod simd {
    use std::arch::aarch64::*;

    /// NEON kernel: 4-lane f32 select-based reconstruction + FMA, 4 dims/iter.
    /// Falls back to scalar for the remainder if dim % 4 != 0.
    pub unsafe fn score_neon(packed: &[u8], centroids: &[f32], query: &[f32], dim: usize) -> f32 {
        let c = [
            vdupq_n_f32(centroids[0]),
            vdupq_n_f32(centroids[1]),
            vdupq_n_f32(centroids[2]),
            vdupq_n_f32(centroids[3]),
        ];
        let mut acc = vdupq_n_f32(0.0);
        let mut i = 0;

        while i + 4 <= dim {
            let codes: [u32; 4] = std::array::from_fn(|j| {
                let idx = i + j;
                ((packed[idx / 4] >> ((idx % 4) * 2)) & 0b11) as u32
            });
            let code_v = vld1q_u32(codes.as_ptr());
            let mut recon = c[3];
            recon = vbslq_f32(vceqq_u32(code_v, vdupq_n_u32(2)), c[2], recon);
            recon = vbslq_f32(vceqq_u32(code_v, vdupq_n_u32(1)), c[1], recon);
            recon = vbslq_f32(vceqq_u32(code_v, vdupq_n_u32(0)), c[0], recon);

            let q = vld1q_f32(query.as_ptr().add(i));
            acc = vfmaq_f32(acc, recon, q);
            i += 4;
        }

        let mut sum = vaddvq_f32(acc);
        while i < dim {
            let bucket = (packed[i / 4] >> ((i % 4) * 2)) & 0b11;
            sum += centroids[bucket as usize] * query[i];
            i += 1;
        }
        sum
    }
}

fn score(packed: &[u8], centroids: &[f32], query: &[f32], dim: usize) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { simd::score_neon(packed, centroids, query, dim) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        score_scalar(packed, centroids, query, dim)
    }
}

#[derive(Serialize, Deserialize)]
pub struct IdMapIndex {
    dim: usize,
    rotation: Vec<f32>,
    boundaries: Vec<f32>,
    centroids: Vec<f32>,
    fitted: bool,
    ids: Vec<u64>,
    packed: Vec<Vec<u8>>,
    correction: Vec<f32>,
}

impl IdMapIndex {
    pub fn new(dim: usize, seed: u64) -> Self {
        IdMapIndex {
            dim,
            rotation: generate_rotation_matrix(dim, seed),
            boundaries: vec![],
            centroids: vec![],
            fitted: false,
            ids: vec![],
            packed: vec![],
            correction: vec![],
        }
    }

    pub fn add_with_ids(&mut self, vectors: &[f32], ids: &[u64]) {
        let n = ids.len();

        if !self.fitted {
            let mut all_rotated = Vec::with_capacity(n * self.dim);
            for i in 0..n {
                let v = &vectors[i * self.dim..(i + 1) * self.dim];
                let (unit, _) = normalize(v);
                all_rotated.extend(apply_rotation(&unit, &self.rotation, self.dim));
            }
            let (b, c) = lloyd_max_quantizer(&all_rotated, 4, 50);
            self.boundaries = b;
            self.centroids = c;
            self.fitted = true;
        }

        for i in 0..n {
            let v = &vectors[i * self.dim..(i + 1) * self.dim];
            let (unit, norm) = normalize(v);
            let rotated = apply_rotation(&unit, &self.rotation, self.dim);

            let bucket_idx: Vec<u8> = rotated.iter().map(|&x| quantize_value(x, &self.boundaries)).collect();
            let recon: Vec<f32> = bucket_idx.iter().map(|&b| self.centroids[b as usize]).collect();

            let dot_u_recon = unit.iter().zip(&recon).map(|(a, b)| a * b).sum::<f32>().max(1e-3);
            let correction = (norm / dot_u_recon).clamp(0.5, 2.0);

            self.packed.push(pack_2bit(&bucket_idx));
            self.ids.push(ids[i]);
            self.correction.push(correction);
        }
    }

    pub fn search(&self, query: &[f32], k: usize) -> (Vec<f32>, Vec<u64>) {
        use rayon::prelude::*;

        let (q_unit, _) = normalize(query);
        let q_rotated = apply_rotation(&q_unit, &self.rotation, self.dim);

        let mut scored: Vec<(f32, u64)> = self.packed.par_iter().zip(&self.ids).zip(&self.correction)
            .map(|((packed, &id), &corr)| {
                (score(packed, &self.centroids, &q_rotated, self.dim) * corr, id)
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        scored.truncate(k);
        scored.into_iter().unzip()
    }

    pub fn remove(&mut self, id: u64) {
        if let Some(pos) = self.ids.iter().position(|&x| x == id) {
            self.ids.remove(pos);
            self.packed.remove(pos);
            self.correction.remove(pos);
        }
    }

    pub fn write(&self, path: &str) -> std::io::Result<()> {
        let file = File::create(path)?;
        bincode::serialize_into(BufWriter::new(file), self).unwrap();
        Ok(())
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let file = File::open(path)?;
        Ok(bincode::deserialize_from(BufReader::new(file)).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_preserves_norm() {
        let dim = 16;
        let rotation = generate_rotation_matrix(dim, 42);
        let v: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
        let (unit, _) = normalize(&v);
        let rotated = apply_rotation(&unit, &rotation, dim);
        let n: f32 = rotated.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-4);
    }

    #[test]
    fn pack_unpack_2bit_roundtrip() {
        let original: Vec<u8> = vec![0, 1, 2, 3, 1, 0, 3, 2, 1];
        let packed = pack_2bit(&original);
        assert_eq!(original, unpack_2bit(&packed, original.len()));
    }

    #[test]
    fn simd_matches_scalar() {
        let dim = 384;
        let centroids = vec![-0.13f32, -0.04, 0.04, 0.13];
        let bucket_idx: Vec<u8> = (0..dim).map(|i| (i % 4) as u8).collect();
        let packed = pack_2bit(&bucket_idx);
        let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.017).sin()).collect();

        let expected = score_scalar(&packed, &centroids, &query, dim);
        let actual = score(&packed, &centroids, &query, dim);
        assert!((expected - actual).abs() < 1e-3, "expected={}, actual={}", expected, actual);
    }

    #[test]
    fn simd_matches_scalar_nonmultiple_of_4() {
        let dim = 385; // exercises the scalar remainder tail
        let centroids = vec![-0.13f32, -0.04, 0.04, 0.13];
        let bucket_idx: Vec<u8> = (0..dim).map(|i| (i % 4) as u8).collect();
        let packed = pack_2bit(&bucket_idx);
        let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.023).cos()).collect();

        let expected = score_scalar(&packed, &centroids, &query, dim);
        let actual = score(&packed, &centroids, &query, dim);
        assert!((expected - actual).abs() < 1e-3);
    }

    #[test]
    fn index_self_query_returns_self() {
        let dim = 32;
        let n = 50;
        let mut rng = StdRng::seed_from_u64(7);
        let mut vectors = vec![0.0f32; n * dim];
        for x in vectors.iter_mut() {
            *x = StandardNormal.sample(&mut rng);
        }
        let ids: Vec<u64> = (0..n as u64).collect();
        let mut idx = IdMapIndex::new(dim, 42);
        idx.add_with_ids(&vectors, &ids);
        let query = &vectors[10 * dim..11 * dim];
        let (_, result_ids) = idx.search(query, 5);
        assert!(result_ids.contains(&10));
    }
}

// ---- Python bindings ----

use pyo3::prelude::*;

#[pyclass(name = "IdMapIndex")]
pub struct PyIdMapIndex {
    inner: IdMapIndex,
}

#[pymethods]
impl PyIdMapIndex {
    #[new]
    fn new(dim: usize, seed: u64) -> Self {
        PyIdMapIndex { inner: IdMapIndex::new(dim, seed) }
    }

    fn add_with_ids(&mut self, vectors: Vec<f32>, ids: Vec<u64>) {
        self.inner.add_with_ids(&vectors, &ids);
    }

    fn search(&self, query: Vec<f32>, k: usize) -> (Vec<f32>, Vec<u64>) {
        self.inner.search(&query, k)
    }

    fn remove(&mut self, id: u64) {
        self.inner.remove(id);
    }

    fn write(&self, path: &str) -> PyResult<()> {
        self.inner.write(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = IdMapIndex::load(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(PyIdMapIndex { inner })
    }
}

#[pymodule]
fn turbovec_lite_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIdMapIndex>()?;
    Ok(())
}
