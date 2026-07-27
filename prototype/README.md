This folder isn't a polished, standalone project — it's the NumPy groundwork
that came before [`turbovec_lite_rs`](../core), the Rust port. I wanted to be
sure I actually understood Google Research's **TurboQuant** algorithm before
writing a single line of Rust, so everything here exists to answer one
question: *does this compression scheme actually work, and how well?*

It's kept here for the record, mostly so the story of how the algorithm was
validated — including a real bug that got caught along the way — isn't lost
once the Rust version became the "real" implementation.

## What's in here

| File | What it does |
|------|---------------|
| `prototype.py` | The algorithm itself: normalize, rotate, Lloyd-Max quantize, bit-pack, brute-force search, recall scoring. Runs as a script with six numbered stages, each printing a sanity check. |
| `prototype_real_embeddings.py` | The same pipeline run against real sentence embeddings instead of random vectors, to check whether the algorithm holds up on data with actual semantic structure. |
| `turbovec_lite.py` | A cleaned-up, class-based version of the same logic — `IdMapIndex`, with `add_with_ids`, `search`, `remove`, and `write`/`load` persistence. This is the version that got translated into Rust. |
| `test_index.py` | Exercises `IdMapIndex`: self-query correctness, persistence round-trip, delete. |

## What got validated, and how

**Rotation preserves geometry.** Before trusting the quantizer, I checked
that the random rotation step doesn't distort anything it shouldn't:
vector norms stay at 1.0, and dot products between pairs of vectors are
identical before and after rotation (`prototype.py`, Stage 2). This matters
because the whole point of the rotation is to make each coordinate
statistically predictable *without* changing the actual geometry the search
depends on.

**Quantization and packing round-trip cleanly.** `pack_2bit` /
`unpack_2bit` are checked against each other directly — pack, then unpack,
and assert the bucket indices come back unchanged (Stage 4). Compression
came out at 32x on the first run rather than the expected 16x, because
NumPy had silently promoted the rotated vectors to `float64`; worth knowing
about, since it means the "true" packed-vs-raw ratio should be judged
against `float32`, not whatever NumPy defaults to.

**Recall was measured properly, twice.** The first pass at measuring
recall on real sentence embeddings came back much worse than expected
(0.25 at 2-bit) — not because the algorithm was broken, but because the
test sentences had too many near-duplicates, so "top-10 nearest neighbors"
was mostly picking among ties that quantization noise could trivially
reshuffle. `prototype_real_embeddings.py` uses 800 sentences built from
20 subjects × 20 actions (560 of them genuinely unique) specifically to
avoid that trap.

**A real bug: the correction factor could go negative.** `IdMapIndex`
applies a length-renormalization correction to compensate for
quantization's tendency to underestimate inner products
(`correction = norm / dot(unit, reconstruction)`). In the first version,
when a reconstructed vector happened to land nearly orthogonal to the
original, that denominator went close to zero — and the correction could
blow up to values in the thousands, or flip negative and invert the
similarity ranking outright. Diagnosed by comparing a known vector's search
rank with and without the correction applied (rank 0 without it, rank 383
out of 500 with it), and fixed by clamping the correction to `[0.5, 2.0]`
in `turbovec_lite.py`. That fix is the reason `test_index.py`'s self-query
test — searching with a vector that's already in the index and expecting
it back at the top — actually passes.

## Results

Measured in `prototype_real_embeddings.py`, real sentence embeddings
(`sentence-transformers`, `all-MiniLM-L6-v2`, 384 dimensions), 800
sentences, recall against exact (uncompressed) search:

| Bit width | Recall@10 | Bytes / vector | Compression |
|-----------|-----------|-----------------|--------------|
| 2-bit     | 0.84      | 96              | 16.0x        |
| 4-bit     | 0.97      | 192             | 8.0x         |

For comparison, the same pipeline on pure random Gaussian vectors — no real
structure to exploit — landed at 0.65 (2-bit) and 0.90 (4-bit). The gap
between the two is the whole reason real embeddings compress better than
noise: TurboQuant is exploiting actual semantic structure, not just
throwing away precision uniformly.

## Where this went next

Once `turbovec_lite.py`'s `IdMapIndex` was passing its correctness tests
here, the same logic — normalize, rotate, quantize, pack, correct, search —
was ported into Rust as the `IdMapIndex` struct in
[`turbovec_lite_rs`](../core), which is where the SIMD and
performance work happened. See that project's README for the Rust side of
the story, including a re-run of this same recall benchmark through the
compiled Rust/Python module (0.840 — matching this prototype's 0.838 almost
exactly).
