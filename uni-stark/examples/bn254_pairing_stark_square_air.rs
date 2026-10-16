use std::time::Instant;

use ark_bn254::{Config, Fq, Fq2, Fq6, Fq12, G1Affine, G2Affine};
use ark_ec::{
    AffineRepr,
    bn::{BnConfig, G2Prepared},
};
use ark_ff::{BigInteger, Field, One, PrimeField, Zero};

use num_bigint::{BigInt, BigUint, Sign};

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_challenger::{HashChallenger, SerializingChallenger64};
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_goldilocks::Goldilocks;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_sha256::Sha256;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use p3_uni_stark::{StarkConfig, prove, verify};

type Val = Goldilocks;
type Challenge = BinomialExtensionField<Val, 2>;

// ============================================================
// BN254 base-field / limb parameters
// ============================================================

const LIMBS: usize = 16;
const LIMB_BITS: usize = 16;
const BASE: u64 = 1 << LIMB_BITS;

const FQ_COUNT_IN_FQ12: usize = 12;
const FQ12_LIMBS: usize = FQ_COUNT_IN_FQ12 * LIMBS;

// q in little-endian base 2^16.
const Q_LIMBS: [u64; LIMBS] = [
    64839, 55420, 35862, 15392, 51853, 26737, 27281, 38785, 22621, 33153, 17846, 47184, 41001,
    57649, 20082, 12388,
];

// ============================================================
// Operation selectors.
//
// Row i describes the operation mapping ACC_i -> ACC_{i+1}.
// ============================================================

const SEL_SQUARE: usize = 0;
const SEL_DOUBLE: usize = 1;
const SEL_ADD: usize = 2;
const SEL_FINAL: usize = 3;
const SEL_PAD: usize = 4;
const NUM_SELECTORS: usize = 5;

// ============================================================
// Metadata
// ============================================================

const COEFF_INDEX: usize = NUM_SELECTORS;
const ATE_INDEX: usize = COEFF_INDEX + 1;
const ATE_DIGIT: usize = ATE_INDEX + 1;

// ============================================================
// Fq12 accumulator
// ============================================================

const ACC_START: usize = ATE_DIGIT + 1;
const ACC_END: usize = ACC_START + FQ12_LIMBS;

// ============================================================
// Evaluated sparse line: c0,c3,c4 in Fq2 => 96 limbs
// ============================================================

const LINE_FQ_COUNT: usize = 6;
const LINE_START: usize = ACC_END;
const LINE_END: usize = LINE_START + LINE_FQ_COUNT * LIMBS;

// ============================================================
// Range bits for all accumulator limbs.
//
// This makes the 16-bit limb interpretation sound.
// ============================================================

const ACC_BITS_START: usize = LINE_END;
const ACC_BITS_COUNT: usize = FQ12_LIMBS * LIMB_BITS;
const ACC_BITS_END: usize = ACC_BITS_START + ACC_BITS_COUNT;

// ============================================================
// Square modular-reduction witnesses.
//
// For each of the 12 output Fq coordinates:
//
//   S = out + k*q
//
// S can be signed, so k can be signed.
//
// Quotient magnitude: 17 limbs x 16 bits.
// Bound: |k| < roughly 1300*q for this Fq12 square formula,
// so 272 bits is plenty.
//
// Each limb equation has a signed carry. We store 31 internal
// carries (carry_1 ... carry_31); carry_0=carry_32=0.
//
// A 40-bit carry magnitude is comfortably above the concrete
// convolution bound for this formula.
// ============================================================

const SQ_Q_LIMBS: usize = 17;
const SQ_CARRY_COUNT: usize = 31;
const SQ_CARRY_BITS: usize = 40;

const SQ_Q_SIGN_START: usize = ACC_BITS_END;
const SQ_Q_SIGN_END: usize = SQ_Q_SIGN_START + FQ_COUNT_IN_FQ12;

const SQ_Q_BITS_START: usize = SQ_Q_SIGN_END;
const SQ_Q_BITS_COUNT: usize = FQ_COUNT_IN_FQ12 * SQ_Q_LIMBS * LIMB_BITS;
const SQ_Q_BITS_END: usize = SQ_Q_BITS_START + SQ_Q_BITS_COUNT;

const SQ_CARRY_SIGN_START: usize = SQ_Q_BITS_END;
const SQ_CARRY_SIGN_COUNT: usize = FQ_COUNT_IN_FQ12 * SQ_CARRY_COUNT;
const SQ_CARRY_SIGN_END: usize = SQ_CARRY_SIGN_START + SQ_CARRY_SIGN_COUNT;

const SQ_CARRY_BITS_START: usize = SQ_CARRY_SIGN_END;
const SQ_CARRY_BITS_COUNT: usize = FQ_COUNT_IN_FQ12 * SQ_CARRY_COUNT * SQ_CARRY_BITS;
const SQ_CARRY_BITS_END: usize = SQ_CARRY_BITS_START + SQ_CARRY_BITS_COUNT;

const WIDTH: usize = SQ_CARRY_BITS_END;

// ============================================================
// Operation
// ============================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    Square,
    DoubleLine,
    AddLine,
    FinalLine,
    Pad,
}

// ============================================================
// Native schedule row
// ============================================================

#[derive(Clone)]
struct NativeRow {
    op: Op,
    coeff_index: usize,
    ate_index: usize,
    ate_digit: i8,
    accumulator: Fq12,
    line: Option<(Fq2, Fq2, Fq2)>,
}

// ============================================================
// Fq12 coordinate helpers
// ============================================================

fn fq12_to_coords(x: &Fq12) -> [Fq; 12] {
    [
        x.c0.c0.c0, x.c0.c0.c1, x.c0.c1.c0, x.c0.c1.c1, x.c0.c2.c0, x.c0.c2.c1, x.c1.c0.c0,
        x.c1.c0.c1, x.c1.c1.c0, x.c1.c1.c1, x.c1.c2.c0, x.c1.c2.c1,
    ]
}

fn coords_to_fq12(a: &[Fq; 12]) -> Fq12 {
    Fq12::new(
        Fq6::new(
            Fq2::new(a[0], a[1]),
            Fq2::new(a[2], a[3]),
            Fq2::new(a[4], a[5]),
        ),
        Fq6::new(
            Fq2::new(a[6], a[7]),
            Fq2::new(a[8], a[9]),
            Fq2::new(a[10], a[11]),
        ),
    )
}

// Coordinate basis is u^u v^v w^w with:
//
//   u^2 = -1
//   v^3 = 9 + u
//   w^2 = v
//
// index = w*6 + v*2 + u.
fn basis_decode(index: usize) -> (usize, usize, usize) {
    let w = index / 6;
    let rem = index % 6;
    let v = rem / 2;
    let u = rem % 2;
    (u, v, w)
}

fn basis_encode(u: usize, v: usize, w: usize) -> usize {
    w * 6 + v * 2 + u
}

// Returns reduced output basis terms (coordinate, small signed coeff).
fn basis_product(i: usize, j: usize) -> Vec<(usize, i64)> {
    let (u1, v1, w1) = basis_decode(i);
    let (u2, v2, w2) = basis_decode(j);

    let mut coeff: i64 = 1;
    let mut u = u1 + u2;
    let mut v = v1 + v2;
    let mut w = w1 + w2;

    if u >= 2 {
        u -= 2;
        coeff = -coeff;
    }

    if w >= 2 {
        w -= 2;
        v += 1;
    }

    if v < 3 {
        return vec![(basis_encode(u, v, w), coeff)];
    }

    v -= 3;

    if u == 0 {
        vec![
            (basis_encode(0, v, w), 9 * coeff),
            (basis_encode(1, v, w), coeff),
        ]
    } else {
        vec![
            (basis_encode(0, v, w), -coeff),
            (basis_encode(1, v, w), 9 * coeff),
        ]
    }
}

fn add_scaled_fq(acc: &mut Fq, x: Fq, coeff: i64) {
    let mag = coeff.unsigned_abs();
    let scaled = x * Fq::from(mag);

    if coeff >= 0 {
        *acc += scaled;
    } else {
        *acc -= scaled;
    }
}

fn fq12_mul_direct(a: &Fq12, b: &Fq12) -> Fq12 {
    let aa = fq12_to_coords(a);
    let bb = fq12_to_coords(b);
    let mut out = [Fq::zero(); 12];

    for i in 0..12 {
        for j in 0..12 {
            let prod = aa[i] * bb[j];

            for (k, coeff) in basis_product(i, j) {
                add_scaled_fq(&mut out[k], prod, coeff);
            }
        }
    }

    coords_to_fq12(&out)
}

fn fq12_square_direct(a: &Fq12) -> Fq12 {
    fq12_mul_direct(a, a)
}

// ============================================================
// Big-int / limb helpers for witness generation
// ============================================================

fn fq_to_biguint(x: &Fq) -> BigUint {
    BigUint::from_bytes_le(&x.into_bigint().to_bytes_le())
}

fn modulus_biguint() -> BigUint {
    BigUint::from_bytes_le(&Fq::MODULUS.to_bytes_le())
}

fn biguint_to_limbs<const N: usize>(x: &BigUint) -> [u64; N] {
    let bytes = x.to_bytes_le();
    let mut out = [0u64; N];

    for i in 0..N {
        let lo = bytes.get(2 * i).copied().unwrap_or(0) as u64;
        let hi = bytes.get(2 * i + 1).copied().unwrap_or(0) as u64;
        out[i] = lo | (hi << 8);
    }

    // Make sure N limbs were enough.
    assert!(
        bytes.len() <= 2 * N,
        "integer does not fit into {} 16-bit limbs",
        N
    );

    out
}

fn fq_to_limbs(x: &Fq) -> [u64; LIMBS] {
    biguint_to_limbs::<LIMBS>(&fq_to_biguint(x))
}

fn signed_bigint_abs_limbs<const N: usize>(x: &BigInt) -> (bool, [u64; N]) {
    let negative = x.sign() == Sign::Minus;

    let abs = match x.sign() {
        Sign::Minus => (-x).to_biguint().unwrap(),
        Sign::NoSign => BigUint::from(0u64),
        Sign::Plus => x.to_biguint().unwrap(),
    };

    (negative, biguint_to_limbs::<N>(&abs))
}

fn limbs_to_bits<const N: usize>(limbs: &[u64; N]) -> Vec<u64> {
    let mut bits = Vec::with_capacity(N * LIMB_BITS);

    for &limb in limbs {
        assert!(limb < BASE);

        for b in 0..LIMB_BITS {
            bits.push((limb >> b) & 1);
        }
    }

    bits
}

fn signed_i128_bits(x: i128, bit_count: usize) -> (bool, Vec<u64>) {
    let negative = x < 0;
    let mag: u128 = x.unsigned_abs();

    assert!(
        mag < (1u128 << bit_count),
        "carry magnitude {} does not fit in {} bits",
        mag,
        bit_count
    );

    let mut bits = Vec::with_capacity(bit_count);

    for b in 0..bit_count {
        bits.push(((mag >> b) & 1) as u64);
    }

    (negative, bits)
}

// ============================================================
// Integer expression for one Fq12 square output coordinate.
//
// S_r = sum_{i,j} coeff(r,i,j) * A_i * A_j
// ============================================================

fn square_integer_output(input: &[BigUint; 12], out_coord: usize) -> BigInt {
    let mut s = BigInt::from(0);

    for i in 0..12 {
        for j in 0..12 {
            let prod = BigInt::from_biguint(Sign::Plus, &input[i] * &input[j]);

            for (k, coeff) in basis_product(i, j) {
                if k == out_coord {
                    s += prod.clone() * coeff;
                }
            }
        }
    }

    s
}

// ============================================================
// Generate all square witnesses for one SQUARE row.
// ============================================================

struct SquareWitness {
    q_signs: [bool; 12],
    q_limbs: [[u64; SQ_Q_LIMBS]; 12],
    carry_signs: [[bool; SQ_CARRY_COUNT]; 12],
    carry_mags: [[u64; SQ_CARRY_COUNT]; 12],
}

fn build_square_witness(input: &Fq12, output: &Fq12) -> SquareWitness {
    let in_coords_fq = fq12_to_coords(input);
    let out_coords_fq = fq12_to_coords(output);

    let in_big: [BigUint; 12] = std::array::from_fn(|i| fq_to_biguint(&in_coords_fq[i]));

    let q_big = modulus_biguint();

    let mut q_signs = [false; 12];
    let mut q_limbs = [[0u64; SQ_Q_LIMBS]; 12];
    let mut carry_signs = [[false; SQ_CARRY_COUNT]; 12];
    let mut carry_mags = [[0u64; SQ_CARRY_COUNT]; 12];

    for r in 0..12 {
        let s = square_integer_output(&in_big, r);
        let out_big = fq_to_biguint(&out_coords_fq[r]);
        let out_int = BigInt::from_biguint(Sign::Plus, out_big.clone());
        let q_int = BigInt::from_biguint(Sign::Plus, q_big.clone());

        let diff = &s - &out_int;

        assert_eq!(
            &diff % &q_int,
            BigInt::from(0),
            "square output coordinate {} is not congruent mod q",
            r
        );

        let quotient = diff / &q_int;

        let (q_neg, q_abs_limbs) = signed_bigint_abs_limbs::<SQ_Q_LIMBS>(&quotient);

        q_signs[r] = q_neg;
        q_limbs[r] = q_abs_limbs;

        // ----------------------------------------------------
        // Build signed carry chain for:
        //
        // carry_t + S_t - (k*q)_t - out_t
        //      = BASE * carry_{t+1}
        //
        // t = 0..31
        // carry_0 = carry_32 = 0.
        // ----------------------------------------------------

        let input_limbs: [[u64; LIMBS]; 12] =
            std::array::from_fn(|i| fq_to_limbs(&in_coords_fq[i]));

        let out_limbs = fq_to_limbs(&out_coords_fq[r]);

        let q_sign_factor: i128 = if q_neg { -1 } else { 1 };

        let mut carry: i128 = 0;

        for t in 0..32 {
            let mut limb_sum: i128 = carry;

            // S_t.
            for i in 0..12 {
                for j in 0..12 {
                    let mut structure_coeff: i64 = 0;

                    for (k, coeff) in basis_product(i, j) {
                        if k == r {
                            structure_coeff += coeff;
                        }
                    }

                    if structure_coeff == 0 {
                        continue;
                    }

                    for ai in 0..LIMBS {
                        if t < ai {
                            continue;
                        }

                        let bj = t - ai;

                        if bj < LIMBS {
                            limb_sum += structure_coeff as i128
                                * input_limbs[i][ai] as i128
                                * input_limbs[j][bj] as i128;
                        }
                    }
                }
            }

            // -(k*q)_t.
            for ki in 0..SQ_Q_LIMBS {
                if t < ki {
                    continue;
                }

                let qj = t - ki;

                if qj < LIMBS {
                    limb_sum -= q_sign_factor * q_abs_limbs[ki] as i128 * Q_LIMBS[qj] as i128;
                }
            }

            // -out_t.
            if t < LIMBS {
                limb_sum -= out_limbs[t] as i128;
            }

            assert_eq!(
                limb_sum % BASE as i128,
                0,
                "non-divisible square carry at output {}, limb {}",
                r,
                t
            );

            let next_carry = limb_sum / BASE as i128;

            if t < 31 {
                let (neg, bits) = signed_i128_bits(next_carry, SQ_CARRY_BITS);

                carry_signs[r][t] = neg;

                let mut mag = 0u64;
                for b in 0..SQ_CARRY_BITS {
                    mag |= bits[b] << b;
                }
                carry_mags[r][t] = mag;
            } else {
                assert_eq!(
                    next_carry, 0,
                    "final square carry is non-zero for output {}",
                    r
                );
            }

            carry = next_carry;
        }
    }

    SquareWitness {
        q_signs,
        q_limbs,
        carry_signs,
        carry_mags,
    }
}

// ============================================================
// Pairing schedule
// ============================================================

fn evaluated_line(coeff: &(Fq2, Fq2, Fq2), p: &G1Affine) -> (Fq2, Fq2, Fq2) {
    let mut c0 = coeff.0;
    let mut c3 = coeff.1;
    let c4 = coeff.2;

    c0.mul_assign_by_fp(&p.y);
    c3.mul_assign_by_fp(&p.x);

    (c0, c3, c4)
}

fn build_miller_schedule() -> Vec<NativeRow> {
    let p = G1Affine::generator();
    let q = G2Affine::generator();
    let prepared_q = G2Prepared::<Config>::from(q);

    println!("ATE loop digits       : {}", Config::ATE_LOOP_COUNT.len());
    println!("Prepared line coeffs  : {}", prepared_q.ell_coeffs.len());

    let mut rows = Vec::<NativeRow>::new();
    let mut f = Fq12::one();
    let mut coeff_index = 0usize;
    let mut square_checks = 0usize;

    for i in (1..Config::ATE_LOOP_COUNT.len()).rev() {
        let digit = Config::ATE_LOOP_COUNT[i - 1];

        if i != Config::ATE_LOOP_COUNT.len() - 1 {
            rows.push(NativeRow {
                op: Op::Square,
                coeff_index,
                ate_index: i,
                ate_digit: digit,
                accumulator: f,
                line: None,
            });

            let direct = fq12_square_direct(&f);
            let mut ark = f;
            ark.square_in_place();

            assert_eq!(
                direct, ark,
                "direct Fq12 square formula disagrees with arkworks"
            );

            f = ark;
            square_checks += 1;
        }

        let coeff = prepared_q.ell_coeffs[coeff_index];
        let line = evaluated_line(&coeff, &p);

        rows.push(NativeRow {
            op: Op::DoubleLine,
            coeff_index,
            ate_index: i,
            ate_digit: digit,
            accumulator: f,
            line: Some(line),
        });

        f.mul_by_034(&line.0, &line.1, &line.2);
        coeff_index += 1;

        if digit == 1 || digit == -1 {
            let coeff = prepared_q.ell_coeffs[coeff_index];
            let line = evaluated_line(&coeff, &p);

            rows.push(NativeRow {
                op: Op::AddLine,
                coeff_index,
                ate_index: i,
                ate_digit: digit,
                accumulator: f,
                line: Some(line),
            });

            f.mul_by_034(&line.0, &line.1, &line.2);
            coeff_index += 1;
        }
    }

    if Config::X_IS_NEGATIVE {
        panic!("X_IS_NEGATIVE=true requires a separate constrained inverse row");
    }

    for _ in 0..2 {
        let coeff = prepared_q.ell_coeffs[coeff_index];
        let line = evaluated_line(&coeff, &p);

        rows.push(NativeRow {
            op: Op::FinalLine,
            coeff_index,
            ate_index: 0,
            ate_digit: 0,
            accumulator: f,
            line: Some(line),
        });

        f.mul_by_034(&line.0, &line.1, &line.2);
        coeff_index += 1;
    }

    assert_eq!(coeff_index, prepared_q.ell_coeffs.len());

    // Terminal state.
    rows.push(NativeRow {
        op: Op::Pad,
        coeff_index,
        ate_index: 0,
        ate_digit: 0,
        accumulator: f,
        line: None,
    });

    println!("Consumed coefficients : {}", coeff_index);
    println!("Fq12 square checks     : {}", square_checks);
    println!("Unpadded schedule rows : {}", rows.len());

    rows
}

// ============================================================
// Trace encoding helpers
// ============================================================

fn push_fq(row: &mut Vec<Val>, x: &Fq) {
    for limb in fq_to_limbs(x) {
        row.push(Val::from_u64(limb));
    }
}

fn push_fq2(row: &mut Vec<Val>, x: &Fq2) {
    push_fq(row, &x.c0);
    push_fq(row, &x.c1);
}

fn push_fq12(row: &mut Vec<Val>, x: &Fq12) {
    push_fq2(row, &x.c0.c0);
    push_fq2(row, &x.c0.c1);
    push_fq2(row, &x.c0.c2);
    push_fq2(row, &x.c1.c0);
    push_fq2(row, &x.c1.c1);
    push_fq2(row, &x.c1.c2);
}

fn push_selectors(row: &mut Vec<Val>, op: Op) {
    let active = match op {
        Op::Square => SEL_SQUARE,
        Op::DoubleLine => SEL_DOUBLE,
        Op::AddLine => SEL_ADD,
        Op::FinalLine => SEL_FINAL,
        Op::Pad => SEL_PAD,
    };

    for i in 0..NUM_SELECTORS {
        row.push(if i == active { Val::ONE } else { Val::ZERO });
    }
}

fn push_bits_u64(row: &mut Vec<Val>, value: u64, bit_count: usize) {
    for b in 0..bit_count {
        row.push(Val::from_u64((value >> b) & 1));
    }
}

// ============================================================
// Encode one trace row.
// `next_acc` is needed to build the SQUARE witness.
// ============================================================

fn encode_row(native: &NativeRow, next_acc: &Fq12) -> Vec<Val> {
    let mut row = Vec::with_capacity(WIDTH);

    push_selectors(&mut row, native.op);

    row.push(Val::from_u64(native.coeff_index as u64));
    row.push(Val::from_u64(native.ate_index as u64));

    row.push(match native.ate_digit {
        -1 => -Val::ONE,
        0 => Val::ZERO,
        1 => Val::ONE,
        x => panic!("unexpected Ate digit {}", x),
    });

    push_fq12(&mut row, &native.accumulator);

    match native.line {
        Some((ref c0, ref c3, ref c4)) => {
            push_fq2(&mut row, c0);
            push_fq2(&mut row, c3);
            push_fq2(&mut row, c4);
        }
        None => {
            for _ in 0..(LINE_FQ_COUNT * LIMBS) {
                row.push(Val::ZERO);
            }
        }
    }

    // --------------------------------------------------------
    // Accumulator limb range bits.
    // --------------------------------------------------------
    let acc_coords = fq12_to_coords(&native.accumulator);

    for coord in 0..12 {
        for limb in fq_to_limbs(&acc_coords[coord]) {
            push_bits_u64(&mut row, limb, LIMB_BITS);
        }
    }

    // --------------------------------------------------------
    // SQUARE witness.
    // --------------------------------------------------------
    if native.op == Op::Square {
        let witness = build_square_witness(&native.accumulator, next_acc);

        // Quotient signs.
        for r in 0..12 {
            row.push(if witness.q_signs[r] {
                Val::ONE
            } else {
                Val::ZERO
            });
        }

        // Quotient limbs as bits.
        for r in 0..12 {
            for limb in 0..SQ_Q_LIMBS {
                push_bits_u64(&mut row, witness.q_limbs[r][limb], LIMB_BITS);
            }
        }

        // Carry signs.
        for r in 0..12 {
            for c in 0..SQ_CARRY_COUNT {
                row.push(if witness.carry_signs[r][c] {
                    Val::ONE
                } else {
                    Val::ZERO
                });
            }
        }

        // Carry magnitudes as bits.
        for r in 0..12 {
            for c in 0..SQ_CARRY_COUNT {
                push_bits_u64(&mut row, witness.carry_mags[r][c], SQ_CARRY_BITS);
            }
        }
    } else {
        for _ in SQ_Q_SIGN_START..WIDTH {
            row.push(Val::ZERO);
        }
    }

    assert_eq!(row.len(), WIDTH, "encoded row width mismatch");

    row
}

// ============================================================
// Trace
// ============================================================

fn make_trace() -> (RowMajorMatrix<Val>, usize, usize) {
    let schedule = build_miller_schedule();
    let real_rows = schedule.len();
    let height = real_rows.next_power_of_two();

    let mut values = Vec::with_capacity(WIDTH * height);

    for i in 0..schedule.len() {
        let next_acc = if i + 1 < schedule.len() {
            schedule[i + 1].accumulator
        } else {
            schedule[i].accumulator
        };

        values.extend_from_slice(&encode_row(&schedule[i], &next_acc));
    }

    let terminal = schedule.last().unwrap().clone();

    assert_eq!(terminal.op, Op::Pad);
    assert_eq!(terminal.coeff_index, 87);

    while values.len() / WIDTH < height {
        values.extend_from_slice(&encode_row(&terminal, &terminal.accumulator));
    }

    (RowMajorMatrix::new(values, WIDTH), real_rows, height)
}

// ============================================================
// AIR expression helpers
// ============================================================

fn expr_from_bits<AB: AirBuilder>(row: &[AB::Var], start: usize, bit_count: usize) -> AB::Expr {
    let mut acc = AB::Expr::ZERO;
    let mut weight = AB::Expr::ONE;

    for b in 0..bit_count {
        let bit: AB::Expr = row[start + b].into();

        acc = acc + bit * weight.clone();

        weight = weight.clone() + weight;
    }

    acc
}

fn signed_expr_from_bits<AB: AirBuilder>(
    row: &[AB::Var],
    sign_col: usize,
    bits_start: usize,
    bit_count: usize,
) -> AB::Expr {
    let sign: AB::Expr = row[sign_col].into();

    let mag = expr_from_bits::<AB>(row, bits_start, bit_count);

    // (1 - 2s) * mag
    (AB::Expr::ONE - sign.clone() - sign) * mag
}

fn acc_limb_expr<AB: AirBuilder>(row: &[AB::Var], coord: usize, limb: usize) -> AB::Expr {
    row[ACC_START + coord * LIMBS + limb].into()
}

fn quotient_limb_expr<AB: AirBuilder>(
    row: &[AB::Var],
    output_coord: usize,
    q_limb: usize,
) -> AB::Expr {
    let start = SQ_Q_BITS_START + (output_coord * SQ_Q_LIMBS + q_limb) * LIMB_BITS;

    expr_from_bits::<AB>(row, start, LIMB_BITS)
}

fn signed_carry_expr<AB: AirBuilder>(
    row: &[AB::Var],
    output_coord: usize,
    carry_index: usize,
) -> AB::Expr {
    let sign_col = SQ_CARRY_SIGN_START + output_coord * SQ_CARRY_COUNT + carry_index;

    let bits_start =
        SQ_CARRY_BITS_START + (output_coord * SQ_CARRY_COUNT + carry_index) * SQ_CARRY_BITS;

    signed_expr_from_bits::<AB>(row, sign_col, bits_start, SQ_CARRY_BITS)
}

// ============================================================
// AIR
// ============================================================

struct PairingAir;

impl<F> BaseAir<F> for PairingAir {
    fn width(&self) -> usize {
        WIDTH
    }
}

impl<AB: AirBuilder> Air<AB> for PairingAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let next = main.next_slice();

        // ====================================================
        // Selector validity
        // ====================================================

        let mut selector_sum = AB::Expr::ZERO;

        for i in 0..NUM_SELECTORS {
            let s: AB::Expr = local[i].into();

            builder.assert_zero(s.clone() * (s.clone() - AB::Expr::ONE));

            selector_sum = selector_sum + s;
        }

        builder.assert_one(selector_sum);

        // ====================================================
        // First accumulator = 1
        // ====================================================

        builder.when_first_row().assert_one(local[ACC_START]);

        for i in (ACC_START + 1)..ACC_END {
            builder.when_first_row().assert_zero(local[i]);
        }

        // ====================================================
        // Accumulator limb range checks
        // ====================================================

        for limb_index in 0..FQ12_LIMBS {
            let limb: AB::Expr = local[ACC_START + limb_index].into();

            let bits_start = ACC_BITS_START + limb_index * LIMB_BITS;

            let reconstructed = expr_from_bits::<AB>(local, bits_start, LIMB_BITS);

            builder.assert_eq(limb, reconstructed);

            for b in 0..LIMB_BITS {
                let bit: AB::Expr = local[bits_start + b].into();

                builder.assert_zero(bit.clone() * (bit - AB::Expr::ONE));
            }
        }

        // ====================================================
        // Basic schedule constraints
        // ====================================================

        let square: AB::Expr = local[SEL_SQUARE].into();

        let pad: AB::Expr = local[SEL_PAD].into();

        // SQUARE and PAD carry no line.
        for i in LINE_START..LINE_END {
            builder.when(square.clone()).assert_zero(local[i]);

            builder.when(pad.clone()).assert_zero(local[i]);
        }

        // Ate digit ∈ {-1,0,+1}.
        let digit: AB::Expr = local[ATE_DIGIT].into();

        builder
            .assert_zero(digit.clone() * (digit.clone() - AB::Expr::ONE) * (digit + AB::Expr::ONE));

        let coeff: AB::Expr = local[COEFF_INDEX].into();

        let next_coeff: AB::Expr = next[COEFF_INDEX].into();

        builder
            .when_transition()
            .when(square.clone())
            .assert_eq(next_coeff.clone(), coeff.clone());

        builder
            .when_transition()
            .when(pad.clone())
            .assert_eq(next_coeff.clone(), coeff.clone());

        let dbl: AB::Expr = local[SEL_DOUBLE].into();

        let add: AB::Expr = local[SEL_ADD].into();

        let final_line: AB::Expr = local[SEL_FINAL].into();

        let consumes = dbl + add + final_line;

        builder
            .when_transition()
            .when(consumes)
            .assert_eq(next_coeff, coeff + AB::Expr::ONE);

        let next_pad: AB::Expr = next[SEL_PAD].into();

        builder
            .when_transition()
            .when(pad.clone())
            .assert_one(next_pad);

        for i in ACC_START..ACC_END {
            let a: AB::Expr = local[i].into();

            let b: AB::Expr = next[i].into();

            builder.when_transition().when(pad.clone()).assert_eq(a, b);
        }

        // ====================================================
        // SQUARE witness bit constraints
        // ====================================================

        for r in 0..12 {
            let sign_col = SQ_Q_SIGN_START + r;

            let s: AB::Expr = local[sign_col].into();

            builder
                .when(square.clone())
                .assert_zero(s.clone() * (s - AB::Expr::ONE));

            for ql in 0..SQ_Q_LIMBS {
                let start = SQ_Q_BITS_START + (r * SQ_Q_LIMBS + ql) * LIMB_BITS;

                for b in 0..LIMB_BITS {
                    let bit: AB::Expr = local[start + b].into();

                    builder
                        .when(square.clone())
                        .assert_zero(bit.clone() * (bit - AB::Expr::ONE));
                }
            }

            for c in 0..SQ_CARRY_COUNT {
                let sign_col = SQ_CARRY_SIGN_START + r * SQ_CARRY_COUNT + c;

                let cs: AB::Expr = local[sign_col].into();

                builder
                    .when(square.clone())
                    .assert_zero(cs.clone() * (cs - AB::Expr::ONE));

                let bits_start = SQ_CARRY_BITS_START + (r * SQ_CARRY_COUNT + c) * SQ_CARRY_BITS;

                for b in 0..SQ_CARRY_BITS {
                    let bit: AB::Expr = local[bits_start + b].into();

                    builder
                        .when(square.clone())
                        .assert_zero(bit.clone() * (bit - AB::Expr::ONE));
                }
            }
        }

        // ====================================================
        // ACTUAL NON-NATIVE Fq12 SQUARE CONSTRAINT
        //
        // For each output Fq coordinate r and integer limb t:
        //
        // carry_t
        // + S_r[t]
        // - (k_r*q)[t]
        // - out_r[t]
        // - BASE*carry_{t+1}
        // = 0.
        //
        // S_r is the direct quadratic Fq12 square formula from
        // the tower relations.
        // ====================================================

        for r in 0..12 {
            let q_sign: AB::Expr = local[SQ_Q_SIGN_START + r].into();

            // sign factor = 1 - 2*q_sign
            let q_sign_factor = AB::Expr::ONE - q_sign.clone() - q_sign;

            for t in 0..32 {
                let carry_cur = if t == 0 {
                    AB::Expr::ZERO
                } else {
                    signed_carry_expr::<AB>(local, r, t - 1)
                };

                let carry_next = if t == 31 {
                    AB::Expr::ZERO
                } else {
                    signed_carry_expr::<AB>(local, r, t)
                };

                let mut expr = carry_cur;

                // S_r[t].
                for i in 0..12 {
                    for j in 0..12 {
                        let mut structure_coeff: i64 = 0;

                        for (k, c) in basis_product(i, j) {
                            if k == r {
                                structure_coeff += c;
                            }
                        }

                        if structure_coeff == 0 {
                            continue;
                        }

                        for ai in 0..LIMBS {
                            if t < ai {
                                continue;
                            }

                            let bj = t - ai;

                            if bj >= LIMBS {
                                continue;
                            }

                            let a = acc_limb_expr::<AB>(local, i, ai);

                            let b = acc_limb_expr::<AB>(local, j, bj);

                            let term = a * b;

                            let mag = structure_coeff.unsigned_abs();

                            let scaled = term * AB::Expr::from_u64(mag);

                            if structure_coeff > 0 {
                                expr = expr + scaled;
                            } else {
                                expr = expr - scaled;
                            }
                        }
                    }
                }

                // -(k*q)[t].
                let mut kq = AB::Expr::ZERO;

                for ki in 0..SQ_Q_LIMBS {
                    if t < ki {
                        continue;
                    }

                    let qj = t - ki;

                    if qj >= LIMBS {
                        continue;
                    }

                    let klimb = quotient_limb_expr::<AB>(local, r, ki);

                    kq = kq + klimb * AB::Expr::from_u64(Q_LIMBS[qj]);
                }

                expr = expr - q_sign_factor.clone() * kq;

                // - output limb.
                if t < LIMBS {
                    let out: AB::Expr = next[ACC_START + r * LIMBS + t].into();

                    expr = expr - out;
                }

                // -BASE*carry_next.
                expr = expr - carry_next * AB::Expr::from_u64(BASE);

                builder
                    .when_transition()
                    .when(square.clone())
                    .assert_zero(expr);
            }
        }
    }
}

// ============================================================
// Main
// ============================================================

fn main() {
    println!("==============================================");
    println!(" Plonky3 BN254 Pairing STARK");
    println!(" Miller Loop: Fq12 SQUARE is AIR-constrained");
    println!("==============================================");
    println!();

    println!("Native proving field : Goldilocks");
    println!("Pairing curve        : BN254");
    println!("Accumulator field    : Fq12");
    println!("Fq representation    : 16 x 16-bit limbs");
    println!();

    let (trace, real_rows, trace_height) = make_trace();

    println!();
    println!("==============================================");
    println!(" Trace layout");
    println!("==============================================");
    println!();

    println!("Selectors             : {}", NUM_SELECTORS);
    println!("Base schedule columns : {}", LINE_END);
    println!("Accumulator range bits: {}", ACC_BITS_COUNT);
    println!("Square quotient bits  : {}", SQ_Q_BITS_COUNT);
    println!("Square carry signs    : {}", SQ_CARRY_SIGN_COUNT);
    println!("Square carry bits     : {}", SQ_CARRY_BITS_COUNT);
    println!("Trace width           : {} columns", WIDTH);
    println!("Real schedule rows    : {}", real_rows);
    println!("Padded trace height   : {}", trace_height);
    println!("Trace cells           : {}", WIDTH * trace_height);
    println!();

    // ========================================================
    // Plonky3 config
    // ========================================================

    type ByteHash = Sha256;
    type FieldHash = SerializingHasher<ByteHash>;

    let byte_hash = ByteHash {};
    let field_hash = FieldHash::new(byte_hash);

    type Compress = CompressionFunctionFromHasher<ByteHash, 2, 32>;

    let compress = Compress::new(byte_hash);

    type ValMmcs = MerkleTreeMmcs<Val, u8, FieldHash, Compress, 2, 32>;

    let val_mmcs = ValMmcs::new(field_hash, compress, 3);

    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;

    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    type Dft = Radix2DitParallel<Val>;

    let dft = Dft::default();

    let fri_params = FriParameters::new_benchmark(challenge_mmcs);

    type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;

    let pcs = Pcs::new(dft, val_mmcs, fri_params);

    type Challenger = SerializingChallenger64<Val, HashChallenger<u8, ByteHash, 32>>;

    let challenger = Challenger::from_hasher(vec![], byte_hash);

    type ConfigType = StarkConfig<Pcs, Challenge, Challenger>;

    let config = ConfigType::new(pcs, challenger);

    // ========================================================
    // Prove
    // ========================================================

    let prove_start = Instant::now();

    let proof = prove(&config, &PairingAir, trace, &[]).expect("proof generation failed");

    let proving_time = prove_start.elapsed();

    let proof_bytes = postcard::to_allocvec(&proof).expect("proof serialization failed");

    // ========================================================
    // Verify
    // ========================================================

    let verify_start = Instant::now();

    verify(&config, &PairingAir, &proof, &[]).expect("verification failed");

    let verification_time = verify_start.elapsed();

    println!("==============================================");
    println!(" STARK result");
    println!("==============================================");
    println!();

    println!("Proof size        : {} bytes", proof_bytes.len());
    println!("Proving time      : {:?}", proving_time);
    println!("Verification time : {:?}", verification_time);
    println!();

    println!("SUCCESS:");
    println!("All Miller-loop Fq12 SQUARE transitions are now");
    println!("proved non-natively by the Plonky3 AIR.");
    println!();
    println!("Still required for complete pairing proof:");
    println!("1. constrain mul_by_034 line transitions");
    println!("2. constrain final Miller output");
    println!("3. constrain final exponentiation");
}
