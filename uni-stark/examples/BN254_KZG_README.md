# BN254 Pairing and KZG Verification inside Plonky3 STARK/FRI

## Purpose

This directory contains the implementation and experiments for proving a BN254 pairing relation, and then a KZG opening verification relation, inside a transparent Plonky3 STARK/FRI proof.

The main end-to-end result is a **single monolithic Plonky3 STARK proof** of the KZG pairing equation

\[
e(C-yG_1,G_2)=e(\pi,\tau G_2-zG_2),
\]

for a concrete degree-6 polynomial over the BN254 scalar field.

The current implementation is correctness-first and intentionally unoptimized. It uses non-native BN254 base-field arithmetic inside Plonky3's Goldilocks field, with explicit limb decomposition and carry constraints.

This README is intended as a handoff document for continuing the implementation.

---

# 1. Current Status

The following pieces are working end-to-end:

- BN254 native pairing baseline using arkworks.
- Non-native BN254 \(F_q\) arithmetic over Goldilocks.
- Non-native \(F_{q^2}\) multiplication and squaring.
- Direct tower representation of \(F_{q^{12}}\).
- AIR-constrained Miller-loop \(F_{q^{12}}\) squaring.
- AIR-constrained Miller-loop sparse `mul_by_034` multiplication.
- Exact BN254 Miller-loop schedule and all 87 prepared line coefficients.
- Parameterized BN254 pairing for arbitrary \(P\in G_1\), \(Q\in G_2\).
- Exact BN optimal-Ate final exponentiation represented as a 2980-bit linear exponentiation.
- Complete BN254 pairing proof inside Plonky3.
- Degree-6 KZG verification using two pairing STARKs.
- Monolithic KZG verification using **one STARK proof** containing both pairings.
- Equality of the two final \(F_{q^{12}}\) pairing outputs is constrained inside the AIR.
- Benchmarking of:
  - proof size,
  - proving time,
  - verification time,
  - trace width,
  - trace height,
  - trace cells,
  - AIR constraint count,
  - maximum constraint degree,
  - constraint evaluations,
  - peak resident RAM.

The main file to continue from is:

```text
uni-stark/examples/bn254_kzg_degree6_monolithic.rs
```

---

# 2. Repository / Experimental Tree

The relevant experimental files are organized conceptually as follows:

```text
Plonky3/
└── uni-stark/
    └── examples/
        ├── bn254_pairing_diagnostic.rs
        ├── bn254_pairing_stark_square_air.rs
        ├── bn254_pairing_complete_exact_exp_v2.rs
        ├── bn254_pairing_parameterized.rs
        ├── bn254_kzg_degree6_two_pairings.rs
        ├── bn254_kzg_degree6_metrics.rs
        └── bn254_kzg_degree6_monolithic.rs
```

Recommended interpretation:

```text
bn254_pairing_diagnostic.rs
    |
    |-- checks native Miller loop
    |-- checks final-exponent conventions
    |
    v
bn254_pairing_stark_square_air.rs
    |
    |-- first true AIR-constrained Fq12 Miller square
    |
    v
bn254_pairing_complete_exact_exp_v2.rs
    |
    |-- complete fixed pairing e(G1,G2)
    |
    v
bn254_pairing_parameterized.rs
    |
    |-- pairing generalized to arbitrary P in G1 and Q in G2
    |
    v
bn254_kzg_degree6_two_pairings.rs
    |
    |-- two complete STARK pairing proofs
    |-- host compares final Fq12 outputs
    |
    v
bn254_kzg_degree6_metrics.rs
    |
    |-- same two-proof construction
    |-- adds constraint/degree measurements
    |
    v
bn254_kzg_degree6_monolithic.rs
    |
    `-- CURRENT END-TO-END IMPLEMENTATION
        one STARK proof
        two complete pairings
        final equality constrained inside AIR
```

The monolithic file should be treated as the main continuation point.

---

# 3. Cryptographic Goal

KZG verification checks

\[
e(C-yG_1,G_2)
=
e(\pi,\tau G_2-zG_2).
\]

Equivalently,

\[
e(C-yG_1,G_2)
=
e(\pi,(\tau-z)G_2).
\]

For the current test:

\[
f(X)
=
1+2X+3X^2+4X^3+5X^4+6X^5+7X^6.
\]

The evaluation point is

\[
z=5.
\]

The claimed value is

\[
y=f(5)=131836.
\]

The quotient polynomial is

\[
q(X)
=
\frac{f(X)-y}{X-z}
\]

with coefficients

\[
q(X)
=
26367
+5273X
+1054X^2
+210X^3
+41X^4
+7X^5.
\]

The toy KZG setup parameter used in this experiment is

\[
\tau=11.
\]

Therefore

\[
\tau-z=6.
\]

The concrete pairing equation proved by the final implementation is

\[
e(C-131836G_1,G_2)
=
e(\pi,6G_2).
\]

---

# 4. KZG Instance Construction

For the degree-6 test polynomial:

\[
C=f(\tau)G_1
\]

and

\[
\pi=q(\tau)G_1.
\]

The implementation first checks the scalar identity

\[
f(\tau)-y
=
(\tau-z)q(\tau).
\]

Then it constructs:

```text
P_left  = C - y G1
Q_left  = G2

P_right = pi
Q_right = (tau-z) G2
```

For the current instance:

```text
Q_right = 6 G2
```

Native arkworks pairing verification is performed before constructing the STARK witness:

```text
e(P_left, Q_left) == e(P_right, Q_right)
```

This is only a sanity check; the final monolithic STARK separately proves the pairing arithmetic and equality inside the AIR.

---

# 5. Fields and Representation

Plonky3's native proving field is:

```text
Goldilocks
```

BN254 pairing arithmetic is over the BN254 base field \(F_q\), so it is implemented non-natively.

Each \(F_q\) value is represented using:

```text
16 limbs × 16 bits
```

The BN254 base-field modulus is:

```text
21888242871839275222246405745257275088696311157297823662689037894645226208583
```

Little-endian 16-bit limbs:

```text
[
  64839, 55420, 35862, 15392,
  51853, 26737, 27281, 38785,
  22621, 33153, 17846, 47184,
  41001, 57649, 20082, 12388
]
```

The arithmetic relation used throughout the implementation is of the form

\[
S = c + kq,
\]

where:

- \(S\) is the unreduced integer arithmetic expression,
- \(c\) is the canonical field output witness,
- \(k\) is the reduction quotient,
- \(q\) is the BN254 base-field modulus.

At the limb level, signed carries satisfy

\[
\operatorname{carry}_t
+
S[t]
-
(kq)[t]
-
c[t]
=
2^{16}\operatorname{carry}_{t+1}.
\]

The implementation uses bit decomposition to range-constrain:

- field limbs,
- quotient limbs,
- carry magnitudes,
- carry signs.

---

# 6. BN254 Extension-Field Tower

The implementation uses the arkworks BN254 tower:

\[
F_{q^2}=F_q[u]/(u^2+1),
\]

so

\[
u^2=-1.
\]

Then

\[
F_{q^6}=F_{q^2}[v]/(v^3-(9+u)).
\]

Finally

\[
F_{q^{12}}=F_{q^6}[w]/(w^2-v).
\]

The 12 base-field coordinates are ordered as:

```text
0  = c0.c0.c0
1  = c0.c0.c1

2  = c0.c1.c0
3  = c0.c1.c1

4  = c0.c2.c0
5  = c0.c2.c1

6  = c1.c0.c0
7  = c1.c0.c1

8  = c1.c1.c0
9  = c1.c1.c1

10 = c1.c2.c0
11 = c1.c2.c1
```

A direct basis-reduction helper computes products according to:

```text
u^2 = -1
v^3 = 9 + u
w^2 = v
```

This direct coordinate multiplication is used both to construct witnesses and to define the non-native AIR multiplication equations.

---

# 7. Miller Loop

The BN254 optimal-Ate Miller loop follows arkworks' `Bn254` implementation.

For one pairing:

```text
number of prepared line coefficients = 87
Miller operation rows                 = 150
```

The Miller state starts at:

\[
f_0=1.
\]

The loop consists of:

```text
Fq12 square
Miller line multiplication
optional Miller line multiplication
...
two final Miller line multiplications
```

The line function uses arkworks' D-twist sparse multiplication:

```rust
f.mul_by_034(&c0, &c3, &c4);
```

For a G1 point \(P=(x_P,y_P)\), the prepared line coefficients are evaluated as:

```text
c0 <- ell_0 * y_P
c3 <- ell_1 * x_P
c4 <- ell_2
```

The sparse \(F_{q^{12}}\) multiplier has non-zero coordinates corresponding to:

```text
c0 in Fq12.c0.c0
c3 in Fq12.c1.c0
c4 in Fq12.c1.c1
```

The final implementation constrains both:

```text
f_next = f^2
```

and

```text
f_next = f * sparse_line
```

using the non-native \(F_{q^{12}}\) multiplication AIR.

---

# 8. Final Exponentiation

An important debugging result was that the naive exponent

\[
(q^{12}-1)/r
\]

did **not** reproduce arkworks' BN optimal-Ate pairing output when directly applied to the Miller value produced by this implementation.

A diagnostic confirmed:

```text
Miller matches arkworks: true
```

but:

```text
Full exponent via naive (q^12-1)/r: mismatch
```

The implementation therefore derives the exact single exponent represented by arkworks' BN final-exponentiation map.

For BN parameter \(x\), define:

\[
A=12x^3+6x^2+4x,
\]

\[
B=12x^3+6x^2+6x,
\]

\[
D=12x^3+12x^2+6x+1.
\]

The easy part is:

\[
(q^6-1)(q^2+1).
\]

The hard exponent is represented as:

\[
q^3(A-1)+q^2B+qA+D.
\]

Therefore the exact exponent used in the linear STARK state machine is

\[
E
=
(q^6-1)(q^2+1)
\left(
q^3(A-1)+q^2B+qA+D
\right).
\]

For BN254:

```text
bit length(E) = 2980
```

The final exponentiation is implemented as a right-to-left binary exponentiation state machine.

Each bit round consists of:

```text
conditional accumulator multiplication
base squaring
exponent right shift
```

All \(F_{q^{12}}\) multiplication/squaring arithmetic is AIR-constrained.

---

# 9. Complete Pairing

The first complete fixed pairing experiment proved:

\[
e(G_1,G_2).
\]

The pairing was then parameterized to accept arbitrary:

\[
P\in G_1,\qquad Q\in G_2.
\]

A parameterization test used:

\[
P=3G_1,\qquad Q=5G_2.
\]

Result:

```text
Parameterized native pairing check: PASS
```

and the complete STARK verified successfully.

This parameterization is what enabled KZG verification.

---

# 10. Two-Proof KZG Version

The first KZG construction used two independent complete pairing STARKs:

```text
Proof 1:
e(C-yG1, G2)

Proof 2:
e(pi, (tau-z)G2)
```

The Rust host then compared the two terminal \(F_{q^{12}}\) outputs.

This was useful for validating the pairing parameterization, but the equality itself was not yet part of either AIR.

Measured result:

```text
Left proof size   : 26,636,434 bytes
Right proof size  : 26,639,986 bytes
Total proof bytes : 53,276,420 bytes

Total proving time:
65.768087542 s

Total verification time:
110.579708 ms
```

Per pairing AIR:

```text
Real rows                : 6,113
Padded height            : 8,192
Trace width              : 31,851
Trace cells              : 260,923,392
AIR constraints          : 37,596
Max constraint degree    : 4
Constraint evaluations   : 307,986,432
```

Important terminology:

```text
37,596 = AIR constraints for one pairing AIR

75,192 = sum across two independent pairing AIR instances
```

`75,192` should **not** be called the constraint count of one KZG AIR.

---

# 11. Monolithic KZG STARK

The current final implementation is:

```text
bn254_kzg_degree6_monolithic.rs
```

It places both complete BN254 pairings into one trace.

The high-level trace is:

```text
LEFT PAIRING
------------
Miller loop
Final exponentiation
Left pairing output
        |
        | save/bind to shared Fq12 output register
        v
RIGHT PAIRING
-------------
reset pairing accumulator
Miller loop
Final exponentiation
Right pairing output
        |
        | constrain equality to shared Fq12 register
        v
PAD
```

A persistent shared \(F_{q^{12}}\) register is carried across the full trace.

Both pairing `OUTPUT` rows satisfy:

\[
\text{pairing output}
=
\text{shared output}.
\]

Therefore:

\[
e(C-yG_1,G_2)
=
e(\pi,(\tau-z)G_2)
\]

is constrained **inside the AIR**.

This is the main difference from the earlier two-proof experiment.

---

# 12. Final Monolithic Benchmark

Measured on the current macOS development machine:

```text
Pairings in proof       : 2
Proof count             : 1

Real rows               : 12,225
Padded height           : 16,384

Trace width             : 32,044
Trace cells             : 525,008,896

AIR constraints         : 38,366
Max constraint degree   : 4

Constraint evaluations  : 628,588,544

Proof size              : 26,827,106 bytes

Proving time            : 93.73008125 s
Verification time       : 59.99275 ms
```

Peak RAM from macOS `/usr/bin/time -l`:

```text
11,735,875,584 maximum resident set size
```

This is approximately:

```text
11.74 GB decimal
10.93 GiB
```

The main RAM number to report should be:

```text
Peak resident RAM = ~10.93 GiB
```

macOS also reported:

```text
25,347,868,792 peak memory footprint
```

This is a broader macOS memory-accounting metric and should not be substituted for peak RSS when comparing memory usage across proving systems.

---

# 13. Why the Monolithic AIR Has 38,366 Constraints, Not 75,192

AIR constraints describe reusable polynomial transition identities.

The earlier two-proof construction instantiated the same pairing AIR twice:

```text
37,596 + 37,596 = 75,192
```

This was simply a sum across two independent proofs.

The monolithic trace reuses one arithmetic transition system for both pairings.

Only a small number of additional constraints are required for:

- pairing-instance selection,
- switching from left pairing to right pairing,
- resetting the right pairing state,
- carrying the shared final \(F_{q^{12}}\) output,
- constraining both terminal outputs equal.

Therefore the monolithic AIR has:

```text
38,366 symbolic AIR constraints
```

while the increased work appears primarily in the trace height:

```text
8,192 -> 16,384 padded rows
```

and in the total constraint evaluations:

```text
628,588,544
```

---

# 14. Constraint Measurement

Constraint count is obtained with Plonky3's symbolic AIR evaluation:

```rust
let air_layout =
    AirLayout::from_air::<Val>(&air);

let symbolic_constraints =
    get_symbolic_constraints::<Val, _>(
        &air,
        air_layout,
    );

let num_air_constraints =
    symbolic_constraints.len();
```

Maximum degree:

```rust
let max_constraint_degree =
    get_max_constraint_degree::<Val, _>(
        &air,
        air_layout,
        height,
    );
```

For the final monolithic AIR:

```text
AIR constraints       = 38,366
Max constraint degree = 4
```

Constraint evaluations are reported as:

```text
AIR constraints × padded trace height
```

so:

\[
38366\times16384
=
628588544.
\]

AIR constraints are not directly equivalent to R1CS constraints.

For comparisons with Groth16/gnark, report them as separate notions.

---

# 15. Proof Keys / Trusted Setup

The Plonky3 STARK/FRI layer does not generate Groth16-style proving and verification keys.

Therefore report:

```text
STARK proving key (PK)      : N/A
STARK verification key (VK) : N/A
STARK trusted setup         : none
```

The verifier instead uses:

```text
AIR definition
StarkConfig
PCS / FRI configuration
instance/public constants
proof
```

However, the underlying KZG commitment scheme still uses an SRS.

For this toy experiment:

```text
tau = 11
```

So the correct description is:

```text
Transparent STARK/FRI proof layer.
KZG itself retains its SRS/trusted-setup assumption.
```

Do not claim that FRI removes the KZG SRS.

---

# 16. Dependencies

The implementation uses Plonky3 v0.8.0 and arkworks BN254 libraries.

The relevant arkworks dependencies are:

```toml
ark-bn254 = "0.5"
ark-ec = "0.5"
ark-ff = "0.5"
num-bigint.workspace = true
```

The Plonky3 example uses:

```rust
p3_air
p3_challenger
p3_commit
p3_dft
p3_field
p3_fri
p3_goldilocks
p3_matrix
p3_merkle_tree
p3_sha256
p3_symmetric
p3_uni_stark
```

---

# 17. How to Run the Final Monolithic Experiment

From the Plonky3 repository root:

```bash
RUSTFLAGS="-Ctarget-cpu=native" \
cargo build --release \
-p p3-uni-stark \
--example bn254_kzg_degree6_monolithic
```

Then run the already-built binary:

```bash
/usr/bin/time -l \
./target/release/examples/bn254_kzg_degree6_monolithic \
2>&1 | tee monolithic_kzg_benchmark.txt
```

Separating build and execution ensures compiler memory is not included in the RAM measurement.

To extract peak RSS:

```bash
grep "maximum resident set size" monolithic_kzg_benchmark.txt
```

To convert to GiB:

```bash
awk '/maximum resident set size/ {
    printf "Peak RAM: %.3f GiB\n", $1/1024/1024/1024
}' monolithic_kzg_benchmark.txt
```

Expected successful ending:

```text
SUCCESS:
One Plonky3 STARK proved the complete degree-6 KZG
pairing equation, including equality of the two
terminal Fq12 pairing outputs inside the AIR.
```

---

# 18. Current Performance Characteristics

This implementation prioritizes correctness and transparency over optimization.

The main reason the trace is extremely wide is explicit bit decomposition.

For example, the implementation contains:

- 16-bit decompositions of \(F_q\) limbs,
- quotient bit decompositions,
- signed carry bit decompositions,
- multiple \(F_{q^{12}}\) state registers.

The final monolithic trace has:

```text
32,044 columns
```

and more than:

```text
525 million trace cells
```

This explains the roughly 10.93 GiB peak RSS.

---

# 19. Known Limitations

## 19.1 Correctness-first range checking

Range checks are implemented with explicit bit decomposition.

This is extremely expensive.

A production implementation should replace much of this with:

- lookup-based range checks,
- smaller limb strategies,
- optimized non-native field gadgets,
- shared range-check buses/tables where supported.

## 19.2 Canonical field representations

The current witness uses canonical BN254 field values from arkworks.

The arithmetic relations enforce modular equivalence and limb ranges.

A future generalized implementation should explicitly audit whether all externally supplied \(F_q\) values require an independent:

\[
x<q
\]

canonicality constraint.

## 19.3 Pairing inputs are generated by the host

The current KZG points:

```text
C
pi
C-yG1
(tau-z)G2
```

are computed outside the STARK and then used to construct the pairing trace.

The STARK currently proves the pairing computations for those selected points.

If the intended theorem is to prove **the entire KZG verifier**, including deriving:

```text
C-yG1
tauG2-zG2
```

from public commitment/evaluation/opening inputs, then the following still need to move inside the AIR:

- scalar multiplication in G1,
- scalar multiplication in G2,
- elliptic-curve point subtraction/addition,
- public-input binding.

This is an important next step.

## 19.4 KZG commitment construction is outside the STARK

The current experiment computes:

\[
C=f(\tau)G_1
\]

and

\[
\pi=q(\tau)G_1
\]

natively.

The STARK does not prove polynomial commitment construction.

That is normal for a KZG verifier experiment, but should be stated clearly.

## 19.5 Toy SRS parameter

The current experiment uses:

```text
tau = 11
```

for reproducibility.

This is not a secure production KZG setup.

---

# 20. Recommended Next Implementation Tasks

The next collaborator should continue in the following order.

## Task 1: Refactor the monolithic example into modules

The current file is intentionally self-contained.

Split it into reusable modules such as:

```text
uni-stark/examples/bn254_kzg/
├── mod.rs
├── fq.rs
├── fq2.rs
├── fq12.rs
├── miller.rs
├── pairing.rs
├── kzg.rs
├── trace.rs
└── benchmark.rs
```

Possible library layout:

```text
bn254-stark/
├── src/
│   ├── lib.rs
│   ├── non_native/
│   │   ├── mod.rs
│   │   ├── fq.rs
│   │   ├── fq2.rs
│   │   └── fq12.rs
│   ├── pairing/
│   │   ├── mod.rs
│   │   ├── miller.rs
│   │   └── final_exp.rs
│   ├── kzg/
│   │   ├── mod.rs
│   │   └── verifier.rs
│   └── benchmark.rs
└── examples/
    └── degree6_kzg.rs
```

## Task 2: Make KZG statement inputs explicit/public

The target public statement should eventually contain:

```text
C
pi
z
y
G2
tauG2
```

or the appropriate verifier-key representation.

The AIR should derive and prove:

\[
P_1=C-yG_1,
\]

\[
Q_2=\tau G_2-zG_2,
\]

then prove:

\[
e(P_1,G_2)=e(\pi,Q_2).
\]

## Task 3: Add G1 arithmetic gadgets

Implement:

- point addition,
- point doubling,
- scalar multiplication,
- subtraction,
- curve membership.

Needed for:

\[
C-yG_1.
\]

## Task 4: Add G2 arithmetic gadgets

Implement the corresponding \(F_{q^2}\)-coordinate EC arithmetic needed for:

\[
\tau G_2-zG_2.
\]

If \(\tau G_2\) is part of a fixed KZG verifier key, it can be treated as a verifier-key constant rather than recomputed.

## Task 5: Public-input binding

Current pairing inputs are embedded in the generated AIR/witness path.

Move verifier-visible KZG inputs into Plonky3 public values and constrain the internal limb representation against them.

## Task 6: Optimize final exponentiation

The current 2980-bit exponentiation is deliberately simple.

An optimized version should implement arkworks' easy/hard final-exponent decomposition directly:

- conjugation,
- Frobenius maps,
- cyclotomic squaring,
- exponentiation by BN parameter \(x\).

This should significantly reduce rows and proving cost.

## Task 7: Replace bit-decomposition range checks

This is likely the largest optimization opportunity.

Investigate:

- Plonky3 lookup arguments,
- range-check tables,
- shared buses,
- larger limbs,
- carry lookup strategies.

## Task 8: Reduce trace width

Current width:

```text
32,044
```

is the primary memory bottleneck.

Optimize witness layout and reuse temporary columns across mutually exclusive operation rows.

## Task 9: Benchmark multiple polynomial degrees

The pairing-verification cost itself is almost independent of polynomial degree once \(C,\pi,y,z\) are supplied.

Useful degree experiments:

```text
degree 2
degree 4
degree 6
degree 8
degree 16
degree 32
...
```

If commitment/opening construction is also moved inside the proof, degree will become much more relevant.

## Task 10: Compare proving systems carefully

Current related baselines:

### gnark / Groth16

Previous experiment:

```text
BN254
gnark v0.16.3
gnark-crypto v0.21.0

constraints:
1,008,276 R1CS constraints

Groth16 proof:
196 bytes

proving:
~3.31 s

verification:
~1.4 ms
```

### Plonky2

Previous KZG relation experiment:

```text
proof size:
73,968 bytes

verification:
successful
```

These values should not be compared blindly:

```text
R1CS constraints != AIR constraints
```

For Plonky3, report:

```text
AIR constraint polynomials
trace width
trace height
constraint evaluations
proof size
proving time
verification time
peak RAM
```

---

# 21. Suggested Final Benchmark Table

| Metric | Plonky3 Monolithic KZG |
|---|---:|
| Proof system | STARK/FRI |
| Curve | BN254 |
| Native STARK field | Goldilocks |
| KZG polynomial degree | 6 |
| Pairings proved | 2 |
| Proof count | 1 |
| AIR constraints | 38,366 |
| Max AIR degree | 4 |
| Real trace rows | 12,225 |
| Padded trace height | 16,384 |
| Trace width | 32,044 |
| Trace cells | 525,008,896 |
| Constraint evaluations | 628,588,544 |
| Proof size | 26,827,106 B |
| Proving time | 93.730 s |
| Verification time | 59.993 ms |
| Peak RSS | 10.93 GiB |
| STARK PK | N/A |
| STARK VK | N/A |
| STARK trusted setup | None |
| KZG SRS | Required |

---

# 22. Most Important Files for the Next Developer

If only three files are retained for handoff, keep:

```text
bn254_pairing_parameterized.rs
bn254_kzg_degree6_metrics.rs
bn254_kzg_degree6_monolithic.rs
```

The most important one is:

```text
bn254_kzg_degree6_monolithic.rs
```

because it contains the complete current result.

The earlier files are still useful for debugging because they isolate individual stages.

---

# 23. Development Philosophy Used So Far

The implementation was developed incrementally:

```text
integer limb arithmetic
    ->
Fq modular multiplication
    ->
Fq2 arithmetic
    ->
Fq12 tower arithmetic
    ->
Miller square
    ->
Miller sparse line multiplication
    ->
complete Miller loop
    ->
final exponentiation
    ->
fixed pairing
    ->
parameterized pairing
    ->
two-pairing KZG
    ->
monolithic KZG
```

At each stage, native arkworks arithmetic was used only to construct/check witness values.

The intended soundness criterion was:

```text
do not call something "proved"
unless the corresponding arithmetic relation is in the AIR
```

This distinction is important when extending the implementation.

---

# 24. Quick Start for the Collaborator

Clone/build Plonky3 and ensure the arkworks dependencies are available.

Run the final experiment:

```bash
RUSTFLAGS="-Ctarget-cpu=native" \
cargo build --release \
-p p3-uni-stark \
--example bn254_kzg_degree6_monolithic
```

Then:

```bash
/usr/bin/time -l \
./target/release/examples/bn254_kzg_degree6_monolithic
```

The expected final statement is:

```text
SUCCESS:
One Plonky3 STARK proved the complete degree-6 KZG
pairing equation, including equality of the two
terminal Fq12 pairing outputs inside the AIR.
```

---

# 25. Handoff Summary

The core research question demonstrated by this code is:

> Can a BN254 KZG pairing verification relation be represented and proved using a transparent Plonky3 STARK/FRI proof?

For the current concrete degree-6 instance, the answer is yes.

The implementation proves, inside one STARK:

\[
e(C-yG_1,G_2)
=
e(\pi,(\tau-z)G_2).
\]

The main remaining work is no longer proving that pairings can be represented. The next work is engineering the construction into a complete, reusable, efficient KZG verifier with:

- explicit public inputs,
- in-AIR G1/G2 verifier arithmetic,
- optimized final exponentiation,
- efficient non-native range checks,
- much smaller trace width and RAM usage.
