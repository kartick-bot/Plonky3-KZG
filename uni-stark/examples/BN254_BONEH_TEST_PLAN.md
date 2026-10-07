# BN254 Boneh Fully Integrated STARK — Test Plan

## Scope

Current implementation proves exactly three polynomial-equality statements:

\[
f_i(z_i)=F_i(z_i),\qquad i\in\{1,2,3\}.
\]

The integrated proof binds six public G1 commitments and the aggregate proof
point \(W\), derives \(D_i=C_{f_i}-C_{F_i}\) internally, dynamically evaluates
the BN254 Miller lines against \(D_1,D_2,D_3,W\), computes the shared Miller
product, performs one final exponentiation, and constrains the result to one.

## Tier 1 — Fast unit tests

Run on every build.

1. `polynomial_evaluation_matches_demo_equalities`
   - Each demo `left(z)` equals `right(z)`.

2. `quotient_reconstructs_difference`
   - Construct
     \(\Delta_i=f_i-F_i\),
     \(q_i=\Delta_i/(X-z_i)\).
   - Check
     \(q_i(X)(X-z_i)=\Delta_i(X)\).

3. `vanishing_polynomial_is_zero_on_all_points`
   - Construct \(Z_T(X)=\prod_i(X-z_i)\).
   - Check \(Z_T(z_i)=0\) for all three points.

4. `false_semantic_statement_is_rejected_before_proving`
   - Modify one polynomial so `left(z) != right(z)`.
   - API must return an error before building the large STARK trace.

## Tier 2 — Full end-to-end proof test

Ignored by default because the current prototype has an 8192 × 56098 trace.

`full_integrated_proof_and_public_tamper_tests`

Checks:

- valid three-statement instance proves and verifies;
- trace height is 8192;
- trace width is 56098;
- proof serialization is non-empty;
- all seven public G1 statement points are individually mutated and rejected:
  - \(C_{f_1}\)
  - \(C_{F_1}\)
  - \(C_{f_2}\)
  - \(C_{F_2}\)
  - \(C_{f_3}\)
  - \(C_{F_3}\)
  - \(W\)

Run explicitly:

```bash
RUSTFLAGS="-Ctarget-cpu=native" \
cargo test --release \
-p p3-uni-stark \
--example bn254_boneh_integrated_api \
full_integrated_proof_and_public_tamper_tests \
-- --ignored --nocapture
```

## Tier 3 — Verifier-instance mismatch tests

Add after the proof/verifier object is separated cleanly into `prove()` and
`verify()` APIs.

Reuse one valid proof and verify it against a changed verifier instance:

- wrong \(\gamma\);
- wrong \(z_1\), \(z_2\), or \(z_3\);
- wrong \(Q_1,Q_2,Q_3\);
- wrong \(Q_T\);
- commitments from one instance combined with \(W\) from another.

All must reject.

## Tier 4 — Witness/adversarial AIR tests

These target internal soundness rather than just public-input binding.

Deliberately corrupt one trace witness before proving/evaluating constraints:

- one internal \(D_i\) limb;
- one raw prepared-G2 line coefficient limb;
- one dynamically evaluated `c0 * y_P` product;
- one dynamically evaluated `c3 * x_P` product;
- one Miller accumulator limb;
- one final-exponentiation accumulator limb;
- one modular-reduction quotient/carry witness.

The AIR should become unsatisfied in every case.

## Tier 5 — Input-shape and edge cases

Before calling this production-ready, add tests for:

- constant-vs-constant equality;
- different polynomial degrees;
- leading zero coefficients;
- zero difference polynomial \(f_i=F_i\);
- repeated evaluation points;
- evaluation point zero;
- gamma zero;
- gamma one;
- invalid/identity G1 points if externally accepted;
- canonical Fq limb encoding;
- curve membership and subgroup checks for external points.

## Benchmark regression

Keep a benchmark record for:

- proof size;
- proving time;
- verification time;
- trace width;
- trace height;
- trace cells;
- AIR constraint count;
- max constraint degree;
- peak RSS.

Current fully integrated reference run:

- trace width: 56,098
- trace height: 8,192
- trace cells: 459,554,816
- AIR constraints: 63,427
- max degree: 4
- proof size: 46,809,650 bytes
- proving time: ~70.75 s
- verification time: ~99.5 ms
