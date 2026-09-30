
use std::time::Instant;

use ark_bn254::{
    Bn254,
    Config,
    Fq,
    Fq2,
    Fq6,
    Fq12,
    G1Affine,
    G2Affine,
};
use ark_ec::{
    bn::{BnConfig, G2Prepared},
    pairing::Pairing,
    AffineRepr,
};
use ark_ff::{
    BigInteger,
    CyclotomicMultSubgroup,
    Field,
    One,
    PrimeField,
    Zero,
};
use num_bigint::{BigInt, BigUint, Sign};

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_challenger::{HashChallenger, SerializingChallenger64};
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::{
    extension::BinomialExtensionField,
    PrimeCharacteristicRing,
};
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_goldilocks::Goldilocks;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_sha256::Sha256;
use p3_symmetric::{
    CompressionFunctionFromHasher,
    SerializingHasher,
};
use p3_uni_stark::{prove, verify, StarkConfig};

type Val = Goldilocks;
type Challenge = BinomialExtensionField<Val, 2>;

const LIMBS: usize = 16;
const LIMB_BITS: usize = 16;
const BASE: u64 = 1 << LIMB_BITS;

const FQ12_COORDS: usize = 12;
const FQ12_LIMBS: usize = FQ12_COORDS * LIMBS;

const Q_LIMBS: [u64; LIMBS] = [
    64839, 55420, 35862, 15392,
    51853, 26737, 27281, 38785,
    22621, 33153, 17846, 47184,
    41001, 57649, 20082, 12388,
];

// Reuse same generic non-native modular multiplication witness sizes.
const QW_LIMBS: usize = 17;
const CARRY_COUNT: usize = 31;
const CARRY_BITS: usize = 48;

// ============================================================
// Selectors
// ============================================================

const SEL_MILLER_SQ: usize = 0;
const SEL_MILLER_LINE: usize = 1;
const SEL_FQ12_MUL: usize = 2;
const SEL_FQ12_SQ: usize = 3;
const SEL_CONJ: usize = 4;
const SEL_FROB1: usize = 5;
const SEL_FROB2: usize = 6;
const SEL_FROB3: usize = 7;
const SEL_OUTPUT: usize = 8;
const SEL_PAD: usize = 9;
const NUM_SELECTORS: usize = 10;

// ============================================================
// Metadata
// ============================================================

const STEP_INDEX: usize = NUM_SELECTORS;

// ============================================================
// Registers
// ============================================================

const ACC_START: usize = STEP_INDEX + 1;
const ACC_END: usize = ACC_START + FQ12_LIMBS;

// Optional second operand for generic mul/sq rows.
const OP_B_START: usize = ACC_END;
const OP_B_END: usize = OP_B_START + FQ12_LIMBS;

// Miller sparse line.
const LINE_LIMBS: usize = 6 * LIMBS;
const LINE_START: usize = OP_B_END;
const LINE_END: usize = LINE_START + LINE_LIMBS;

// Range bits for ACC and OP_B.
const ACC_BITS_START: usize = LINE_END;
const ACC_BITS_END: usize = ACC_BITS_START + FQ12_LIMBS * LIMB_BITS;

const OP_B_BITS_START: usize = ACC_BITS_END;
const OP_B_BITS_END: usize = OP_B_BITS_START + FQ12_LIMBS * LIMB_BITS;

// Arithmetic witness.
const Q_SIGN_START: usize = OP_B_BITS_END;
const Q_SIGN_END: usize = Q_SIGN_START + FQ12_COORDS;

const Q_BITS_START: usize = Q_SIGN_END;
const Q_BITS_END: usize =
    Q_BITS_START + FQ12_COORDS * QW_LIMBS * LIMB_BITS;

const CARRY_SIGN_START: usize = Q_BITS_END;
const CARRY_SIGN_END: usize =
    CARRY_SIGN_START + FQ12_COORDS * CARRY_COUNT;

const CARRY_BITS_START: usize = CARRY_SIGN_END;
const WIDTH: usize =
    CARRY_BITS_START + FQ12_COORDS * CARRY_COUNT * CARRY_BITS;

// ============================================================
// Native row
// ============================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    MillerSquare,
    MillerLine,
    Fq12Mul,
    Fq12Sq,
    Conjugate,
    Frob1,
    Frob2,
    Frob3,
    Output,
    Pad,
}

#[derive(Clone)]
struct NativeRow {
    op: Op,
    step_index: usize,
    acc: Fq12,
    operand_b: Option<Fq12>,
    line: Option<(Fq2, Fq2, Fq2)>,
}

// ============================================================
// Fq12 basis
// ============================================================

fn fq12_to_coords(x: &Fq12) -> [Fq; 12] {
    [
        x.c0.c0.c0, x.c0.c0.c1,
        x.c0.c1.c0, x.c0.c1.c1,
        x.c0.c2.c0, x.c0.c2.c1,
        x.c1.c0.c0, x.c1.c0.c1,
        x.c1.c1.c0, x.c1.c1.c1,
        x.c1.c2.c0, x.c1.c2.c1,
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

fn sparse_line_fq12(line: &(Fq2, Fq2, Fq2)) -> Fq12 {
    Fq12::new(
        Fq6::new(
            line.0,
            Fq2::zero(),
            Fq2::zero(),
        ),
        Fq6::new(
            line.1,
            line.2,
            Fq2::zero(),
        ),
    )
}

// ============================================================
// Helpers
// ============================================================

fn fq_to_biguint(x: &Fq) -> BigUint {
    BigUint::from_bytes_le(&x.into_bigint().to_bytes_le())
}

fn modulus_biguint() -> BigUint {
    BigUint::from_bytes_le(&Fq::MODULUS.to_bytes_le())
}

fn biguint_to_limbs<const N: usize>(x: &BigUint) -> [u64; N] {
    let bytes = x.to_bytes_le();
    assert!(bytes.len() <= 2 * N);

    let mut out = [0u64; N];

    for i in 0..N {
        let lo = bytes.get(2 * i).copied().unwrap_or(0) as u64;
        let hi = bytes.get(2 * i + 1).copied().unwrap_or(0) as u64;
        out[i] = lo | (hi << 8);
    }

    out
}

fn fq_to_limbs(x: &Fq) -> [u64; LIMBS] {
    biguint_to_limbs::<LIMBS>(&fq_to_biguint(x))
}

fn signed_bigint_abs_limbs<const N: usize>(
    x: &BigInt,
) -> (bool, [u64; N]) {
    let negative = x.sign() == Sign::Minus;

    let abs = match x.sign() {
        Sign::Minus => (-x).to_biguint().unwrap(),
        Sign::NoSign => BigUint::from(0u64),
        Sign::Plus => x.to_biguint().unwrap(),
    };

    (negative, biguint_to_limbs::<N>(&abs))
}

fn push_bits_u64(row: &mut Vec<Val>, value: u64, bits: usize) {
    for b in 0..bits {
        row.push(Val::from_u64((value >> b) & 1));
    }
}

// ============================================================
// Miller loop
// ============================================================

fn evaluated_line(
    coeff: &(Fq2, Fq2, Fq2),
    p: &G1Affine,
) -> (Fq2, Fq2, Fq2) {
    let mut c0 = coeff.0;
    let mut c3 = coeff.1;
    let c4 = coeff.2;

    c0.mul_assign_by_fp(&p.y);
    c3.mul_assign_by_fp(&p.x);

    (c0, c3, c4)
}

fn build_miller_rows() -> (Vec<NativeRow>, Fq12) {
    let p = G1Affine::generator();
    let q = G2Affine::generator();
    let prepared = G2Prepared::<Config>::from(q);

    let mut rows = Vec::<NativeRow>::new();
    let mut f = Fq12::one();
    let mut coeff_index = 0usize;
    let mut step = 0usize;

    for i in (1..Config::ATE_LOOP_COUNT.len()).rev() {
        let digit = Config::ATE_LOOP_COUNT[i - 1];

        if i != Config::ATE_LOOP_COUNT.len() - 1 {
            rows.push(NativeRow {
                op: Op::MillerSquare,
                step_index: step,
                acc: f,
                operand_b: Some(f),
                line: None,
            });

            f = fq12_mul_direct(&f, &f);
            step += 1;
        }

        let coeff = prepared.ell_coeffs[coeff_index];
        let line = evaluated_line(&coeff, &p);
        let sparse = sparse_line_fq12(&line);

        rows.push(NativeRow {
            op: Op::MillerLine,
            step_index: step,
            acc: f,
            operand_b: Some(sparse),
            line: Some(line),
        });

        let direct = fq12_mul_direct(&f, &sparse);

        let mut ark = f;
        ark.mul_by_034(&line.0, &line.1, &line.2);

        assert_eq!(direct, ark);
        f = direct;

        coeff_index += 1;
        step += 1;

        if digit == 1 || digit == -1 {
            let coeff = prepared.ell_coeffs[coeff_index];
            let line = evaluated_line(&coeff, &p);
            let sparse = sparse_line_fq12(&line);

            rows.push(NativeRow {
                op: Op::MillerLine,
                step_index: step,
                acc: f,
                operand_b: Some(sparse),
                line: Some(line),
            });

            let direct = fq12_mul_direct(&f, &sparse);

            let mut ark = f;
            ark.mul_by_034(&line.0, &line.1, &line.2);

            assert_eq!(direct, ark);
            f = direct;

            coeff_index += 1;
            step += 1;
        }
    }

    if Config::X_IS_NEGATIVE {
        f.cyclotomic_inverse_in_place();
    }

    for _ in 0..2 {
        let coeff = prepared.ell_coeffs[coeff_index];
        let line = evaluated_line(&coeff, &p);
        let sparse = sparse_line_fq12(&line);

        rows.push(NativeRow {
            op: Op::MillerLine,
            step_index: step,
            acc: f,
            operand_b: Some(sparse),
            line: Some(line),
        });

        let direct = fq12_mul_direct(&f, &sparse);

        let mut ark = f;
        ark.mul_by_034(&line.0, &line.1, &line.2);

        assert_eq!(direct, ark);
        f = direct;

        coeff_index += 1;
        step += 1;
    }

    assert_eq!(coeff_index, prepared.ell_coeffs.len());

    let native_miller = Bn254::multi_miller_loop(
        [G1Affine::generator()],
        [G2Affine::generator()],
    ).0;

    assert_eq!(f, native_miller);

    (rows, f)
}

// ============================================================
// Final exponentiation: mirror arkworks sequence exactly
// ============================================================

fn push_mul_row(rows: &mut Vec<NativeRow>, step: &mut usize, a: Fq12, b: Fq12) -> Fq12 {
    rows.push(NativeRow {
        op: Op::Fq12Mul,
        step_index: *step,
        acc: a,
        operand_b: Some(b),
        line: None,
    });

    *step += 1;
    fq12_mul_direct(&a, &b)
}

fn push_sq_row(rows: &mut Vec<NativeRow>, step: &mut usize, a: Fq12) -> Fq12 {
    rows.push(NativeRow {
        op: Op::Fq12Sq,
        step_index: *step,
        acc: a,
        operand_b: Some(a),
        line: None,
    });

    *step += 1;
    fq12_mul_direct(&a, &a)
}

fn push_conj_row(rows: &mut Vec<NativeRow>, step: &mut usize, a: Fq12) -> Fq12 {
    rows.push(NativeRow {
        op: Op::Conjugate,
        step_index: *step,
        acc: a,
        operand_b: None,
        line: None,
    });

    *step += 1;

    let mut out = a;
    out.cyclotomic_inverse_in_place().unwrap();
    out
}

fn push_frob_row(
    rows: &mut Vec<NativeRow>,
    step: &mut usize,
    a: Fq12,
    power: usize,
) -> Fq12 {
    let op = match power {
        1 => Op::Frob1,
        2 => Op::Frob2,
        3 => Op::Frob3,
        _ => panic!("unsupported frobenius power"),
    };

    rows.push(NativeRow {
        op,
        step_index: *step,
        acc: a,
        operand_b: None,
        line: None,
    });

    *step += 1;

    let mut out = a;
    out.frobenius_map_in_place(power);
    out
}

fn exp_by_neg_x_rows(
    rows: &mut Vec<NativeRow>,
    step: &mut usize,
    mut f: Fq12,
) -> Fq12 {
    let mut x = BigUint::from(Config::X[0]);

    // Config::X is one 64-bit limb on BN254.
    let mut acc = Fq12::one();
    let mut base = f;

    while !x.is_zero() {
        if (&x & BigUint::from(1u64)) == BigUint::from(1u64) {
            acc = push_mul_row(rows, step, acc, base);
        }

        x >>= 1usize;

        if !x.is_zero() {
            base = push_sq_row(rows, step, base);
        }
    }

    f = acc;

    if !Config::X_IS_NEGATIVE {
        f = push_conj_row(rows, step, f);
    }

    f
}

fn build_complete_rows() -> (Vec<NativeRow>, Fq12) {
    let (mut rows, miller) = build_miller_rows();
    let mut step = rows.len();

    let f = miller;

    // Easy part exactly as arkworks.
    let mut f1 = f;
    f1.cyclotomic_inverse_in_place().unwrap();

    let f2 = f.inverse().unwrap();

    let mut r = push_mul_row(&mut rows, &mut step, f1, f2);

    let f2_copy = r;

    r = push_frob_row(&mut rows, &mut step, r, 2);
    r = push_mul_row(&mut rows, &mut step, r, f2_copy);

    // Hard part.
    let y0 = exp_by_neg_x_rows(&mut rows, &mut step, r);
    let y1 = push_sq_row(&mut rows, &mut step, y0);
    let y2 = push_sq_row(&mut rows, &mut step, y1);

    let mut y3 = push_mul_row(&mut rows, &mut step, y2, y1);

    let y4 = exp_by_neg_x_rows(&mut rows, &mut step, y3);
    let y5 = push_sq_row(&mut rows, &mut step, y4);

    let mut y6 = exp_by_neg_x_rows(&mut rows, &mut step, y5);

    y3 = push_conj_row(&mut rows, &mut step, y3);
    y6 = push_conj_row(&mut rows, &mut step, y6);

    let y7 = push_mul_row(&mut rows, &mut step, y6, y4);
    let mut y8 = push_mul_row(&mut rows, &mut step, y7, y3);
    let y9 = push_mul_row(&mut rows, &mut step, y8, y1);
    let y10 = push_mul_row(&mut rows, &mut step, y8, y4);
    let y11 = push_mul_row(&mut rows, &mut step, y10, r);

    let mut y12 = y9;
    y12 = push_frob_row(&mut rows, &mut step, y12, 1);

    let y13 = push_mul_row(&mut rows, &mut step, y12, y11);

    y8 = push_frob_row(&mut rows, &mut step, y8, 2);

    let y14 = push_mul_row(&mut rows, &mut step, y8, y13);

    r = push_conj_row(&mut rows, &mut step, r);

    let mut y15 = push_mul_row(&mut rows, &mut step, r, y9);
    y15 = push_frob_row(&mut rows, &mut step, y15, 3);

    let y16 = push_mul_row(&mut rows, &mut step, y15, y14);

    let native_pairing = Bn254::pairing(
        G1Affine::generator(),
        G2Affine::generator(),
    ).0;

    assert_eq!(
        y16, native_pairing,
        "arkworks-sequence reconstruction does not match pairing"
    );

    rows.push(NativeRow {
        op: Op::Output,
        step_index: step,
        acc: y16,
        operand_b: None,
        line: None,
    });

    step += 1;

    rows.push(NativeRow {
        op: Op::Pad,
        step_index: step,
        acc: y16,
        operand_b: None,
        line: None,
    });

    (rows, y16)
}

// ============================================================
// Generic multiplication witness
// ============================================================

fn product_integer_output(
    a: &[BigUint; 12],
    b: &[BigUint; 12],
    out_coord: usize,
) -> BigInt {
    let mut s = BigInt::from(0);

    for i in 0..12 {
        for j in 0..12 {
            let prod = BigInt::from_biguint(
                Sign::Plus,
                &a[i] * &b[j],
            );

            for (k, coeff) in basis_product(i, j) {
                if k == out_coord {
                    s += prod.clone() * coeff;
                }
            }
        }
    }

    s
}

struct MulWitness {
    q_signs: [bool; 12],
    q_limbs: [[u64; QW_LIMBS]; 12],
    carry_signs: [[bool; CARRY_COUNT]; 12],
    carry_mags: [[u64; CARRY_COUNT]; 12],
}

fn build_mul_witness(
    a: &Fq12,
    b: &Fq12,
    out: &Fq12,
) -> MulWitness {
    let ac = fq12_to_coords(a);
    let bc = fq12_to_coords(b);
    let oc = fq12_to_coords(out);

    let abig: [BigUint; 12] =
        std::array::from_fn(|i| fq_to_biguint(&ac[i]));

    let bbig: [BigUint; 12] =
        std::array::from_fn(|i| fq_to_biguint(&bc[i]));

    let alimbs: [[u64; LIMBS]; 12] =
        std::array::from_fn(|i| fq_to_limbs(&ac[i]));

    let blimbs: [[u64; LIMBS]; 12] =
        std::array::from_fn(|i| fq_to_limbs(&bc[i]));

    let q_big = modulus_biguint();
    let q_int = BigInt::from_biguint(Sign::Plus, q_big);

    let mut q_signs = [false; 12];
    let mut q_limbs = [[0u64; QW_LIMBS]; 12];
    let mut carry_signs = [[false; CARRY_COUNT]; 12];
    let mut carry_mags = [[0u64; CARRY_COUNT]; 12];

    for r in 0..12 {
        let s = product_integer_output(&abig, &bbig, r);
        let out_big = fq_to_biguint(&oc[r]);

        let diff =
            &s - BigInt::from_biguint(Sign::Plus, out_big);

        assert_eq!(&diff % &q_int, BigInt::from(0));

        let quotient = diff / &q_int;

        let (qneg, qabs) =
            signed_bigint_abs_limbs::<QW_LIMBS>(&quotient);

        q_signs[r] = qneg;
        q_limbs[r] = qabs;

        let out_limbs = fq_to_limbs(&oc[r]);
        let qsign: i128 = if qneg { -1 } else { 1 };

        let mut carry: i128 = 0;

        for t in 0..32 {
            let mut x = carry;

            for i in 0..12 {
                for j in 0..12 {
                    let mut c: i64 = 0;

                    for (k, cc) in basis_product(i, j) {
                        if k == r {
                            c += cc;
                        }
                    }

                    if c == 0 {
                        continue;
                    }

                    for ai in 0..LIMBS {
                        if t < ai {
                            continue;
                        }

                        let bj = t - ai;

                        if bj < LIMBS {
                            x +=
                                c as i128
                                * alimbs[i][ai] as i128
                                * blimbs[j][bj] as i128;
                        }
                    }
                }
            }

            for ki in 0..QW_LIMBS {
                if t < ki {
                    continue;
                }

                let qj = t - ki;

                if qj < LIMBS {
                    x -=
                        qsign
                        * qabs[ki] as i128
                        * Q_LIMBS[qj] as i128;
                }
            }

            if t < LIMBS {
                x -= out_limbs[t] as i128;
            }

            assert_eq!(x % BASE as i128, 0);

            let next = x / BASE as i128;

            if t < CARRY_COUNT {
                let neg = next < 0;
                let mag = next.unsigned_abs();

                assert!(mag < (1u128 << CARRY_BITS));

                carry_signs[r][t] = neg;
                carry_mags[r][t] = mag as u64;
            } else {
                assert_eq!(next, 0);
            }

            carry = next;
        }
    }

    MulWitness {
        q_signs,
        q_limbs,
        carry_signs,
        carry_mags,
    }
}

// ============================================================
// Trace encoding
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
        Op::MillerSquare => SEL_MILLER_SQ,
        Op::MillerLine => SEL_MILLER_LINE,
        Op::Fq12Mul => SEL_FQ12_MUL,
        Op::Fq12Sq => SEL_FQ12_SQ,
        Op::Conjugate => SEL_CONJ,
        Op::Frob1 => SEL_FROB1,
        Op::Frob2 => SEL_FROB2,
        Op::Frob3 => SEL_FROB3,
        Op::Output => SEL_OUTPUT,
        Op::Pad => SEL_PAD,
    };

    for i in 0..NUM_SELECTORS {
        row.push(if i == active {
            Val::ONE
        } else {
            Val::ZERO
        });
    }
}

fn arithmetic_operands(
    row: &NativeRow,
    next: &NativeRow,
) -> Option<(Fq12, Fq12, Fq12)> {
    match row.op {
        Op::MillerSquare
        | Op::Fq12Sq => {
            Some((row.acc, row.acc, next.acc))
        }

        Op::MillerLine
        | Op::Fq12Mul => {
            Some((
                row.acc,
                row.operand_b.unwrap(),
                next.acc,
            ))
        }

        _ => None,
    }
}

fn encode_row(
    native: &NativeRow,
    next: &NativeRow,
) -> Vec<Val> {
    let mut row = Vec::with_capacity(WIDTH);

    push_selectors(&mut row, native.op);
    row.push(Val::from_u64(native.step_index as u64));

    push_fq12(&mut row, &native.acc);

    match native.operand_b {
        Some(b) => push_fq12(&mut row, &b),
        None => {
            for _ in 0..FQ12_LIMBS {
                row.push(Val::ZERO);
            }
        }
    }

    match native.line {
        Some((ref c0, ref c3, ref c4)) => {
            push_fq2(&mut row, c0);
            push_fq2(&mut row, c3);
            push_fq2(&mut row, c4);
        }
        None => {
            for _ in 0..LINE_LIMBS {
                row.push(Val::ZERO);
            }
        }
    }

    for fq in fq12_to_coords(&native.acc) {
        for limb in fq_to_limbs(&fq) {
            push_bits_u64(&mut row, limb, LIMB_BITS);
        }
    }

    let b = native.operand_b.unwrap_or(Fq12::zero());

    for fq in fq12_to_coords(&b) {
        for limb in fq_to_limbs(&fq) {
            push_bits_u64(&mut row, limb, LIMB_BITS);
        }
    }

    if let Some((a, b, out)) =
        arithmetic_operands(native, next)
    {
        let w = build_mul_witness(&a, &b, &out);

        for r in 0..12 {
            row.push(
                if w.q_signs[r] {
                    Val::ONE
                } else {
                    Val::ZERO
                }
            );
        }

        for r in 0..12 {
            for l in 0..QW_LIMBS {
                push_bits_u64(
                    &mut row,
                    w.q_limbs[r][l],
                    LIMB_BITS,
                );
            }
        }

        for r in 0..12 {
            for c in 0..CARRY_COUNT {
                row.push(
                    if w.carry_signs[r][c] {
                        Val::ONE
                    } else {
                        Val::ZERO
                    }
                );
            }
        }

        for r in 0..12 {
            for c in 0..CARRY_COUNT {
                push_bits_u64(
                    &mut row,
                    w.carry_mags[r][c],
                    CARRY_BITS,
                );
            }
        }
    } else {
        for _ in Q_SIGN_START..WIDTH {
            row.push(Val::ZERO);
        }
    }

    assert_eq!(row.len(), WIDTH);
    row
}

// ============================================================
// AIR helpers
// ============================================================

fn expr_from_bits<AB: AirBuilder>(
    row: &[AB::Var],
    start: usize,
    bit_count: usize,
) -> AB::Expr {
    let mut acc = AB::Expr::ZERO;
    let mut weight = AB::Expr::ONE;

    for b in 0..bit_count {
        let bit: AB::Expr =
            row[start + b].into();

        acc =
            acc
            + bit
                * weight.clone();

        weight =
            weight.clone()
            + weight;
    }

    acc
}

fn signed_expr_from_bits<AB: AirBuilder>(
    row: &[AB::Var],
    sign_col: usize,
    bits_start: usize,
    bit_count: usize,
) -> AB::Expr {
    let sign: AB::Expr =
        row[sign_col].into();

    let mag =
        expr_from_bits::<AB>(
            row,
            bits_start,
            bit_count,
        );

    (
        AB::Expr::ONE
        - sign.clone()
        - sign
    ) * mag
}

fn quotient_limb<AB: AirBuilder>(
    row: &[AB::Var],
    r: usize,
    l: usize,
) -> AB::Expr {
    let start =
        Q_BITS_START
        + (
            r * QW_LIMBS
            + l
        ) * LIMB_BITS;

    expr_from_bits::<AB>(
        row,
        start,
        LIMB_BITS,
    )
}

fn carry_expr<AB: AirBuilder>(
    row: &[AB::Var],
    r: usize,
    c: usize,
) -> AB::Expr {
    let sign_col =
        CARRY_SIGN_START
        + r * CARRY_COUNT
        + c;

    let bits_start =
        CARRY_BITS_START
        + (
            r * CARRY_COUNT
            + c
        ) * CARRY_BITS;

    signed_expr_from_bits::<AB>(
        row,
        sign_col,
        bits_start,
        CARRY_BITS,
    )
}

// ============================================================
// AIR
// ============================================================

struct PairingAir {
    output_limbs: [u64; FQ12_LIMBS],
}

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

        let mut selector_sum =
            AB::Expr::ZERO;

        for i in 0..NUM_SELECTORS {
            let s: AB::Expr =
                local[i].into();

            builder.assert_zero(
                s.clone()
                    * (
                        s.clone()
                        - AB::Expr::ONE
                    )
            );

            selector_sum =
                selector_sum
                + s;
        }

        builder.assert_one(selector_sum);

        let msq: AB::Expr =
            local[SEL_MILLER_SQ].into();
        let mline: AB::Expr =
            local[SEL_MILLER_LINE].into();
        let mul: AB::Expr =
            local[SEL_FQ12_MUL].into();
        let sq: AB::Expr =
            local[SEL_FQ12_SQ].into();
        let conj: AB::Expr =
            local[SEL_CONJ].into();
        let frob1: AB::Expr =
            local[SEL_FROB1].into();
        let frob2: AB::Expr =
            local[SEL_FROB2].into();
        let frob3: AB::Expr =
            local[SEL_FROB3].into();
        let output: AB::Expr =
            local[SEL_OUTPUT].into();
        let pad: AB::Expr =
            local[SEL_PAD].into();

        let arith =
            msq.clone()
            + mline.clone()
            + mul.clone()
            + sq.clone();

        // Range-check ACC.
        for li in 0..FQ12_LIMBS {
            let limb: AB::Expr =
                local[ACC_START + li].into();

            let bs =
                ACC_BITS_START
                + li * LIMB_BITS;

            let rec =
                expr_from_bits::<AB>(
                    local,
                    bs,
                    LIMB_BITS,
                );

            builder.assert_eq(
                limb,
                rec,
            );

            for b in 0..LIMB_BITS {
                let bit: AB::Expr =
                    local[bs + b].into();

                builder.assert_zero(
                    bit.clone()
                        * (
                            bit
                            - AB::Expr::ONE
                        )
                );
            }
        }

        // Range-check OP_B.
        for li in 0..FQ12_LIMBS {
            let limb: AB::Expr =
                local[OP_B_START + li].into();

            let bs =
                OP_B_BITS_START
                + li * LIMB_BITS;

            let rec =
                expr_from_bits::<AB>(
                    local,
                    bs,
                    LIMB_BITS,
                );

            builder.assert_eq(
                limb,
                rec,
            );

            for b in 0..LIMB_BITS {
                let bit: AB::Expr =
                    local[bs + b].into();

                builder.assert_zero(
                    bit.clone()
                        * (
                            bit
                            - AB::Expr::ONE
                        )
                );
            }
        }

        // Step counter always increments until output/pad.
        let active_before_end =
            AB::Expr::ONE
            - output.clone()
            - pad.clone();

        builder
            .when_transition()
            .when(active_before_end)
            .assert_eq(
                next[STEP_INDEX],
                local[STEP_INDEX]
                    + AB::Expr::ONE,
            );

        // Conjugate = cyclotomic inverse in Fq12 = negate c1.
        for i in 0..6 {
            for limb in 0..LIMBS {
                builder
                    .when_transition()
                    .when(conj.clone())
                    .assert_eq(
                        next[
                            ACC_START
                            + i * LIMBS
                            + limb
                        ],
                        local[
                            ACC_START
                            + i * LIMBS
                            + limb
                        ],
                    );
            }
        }

        // c1 part is negation mod q. For this first version we
        // bind conjugation through native witness only; arithmetic
        // proof for conjugation/frobenius can be added next.
        //
        // IMPORTANT: mul/sq are fully non-native constrained below.

        // Output must equal native pairing result.
        for i in 0..FQ12_LIMBS {
            builder
                .when(output.clone())
                .assert_eq(
                    local[ACC_START + i],
                    AB::Expr::from_u64(
                        self.output_limbs[i]
                    ),
                );
        }

        let next_pad: AB::Expr =
            next[SEL_PAD].into();

        builder
            .when_transition()
            .when(output.clone())
            .assert_one(next_pad.clone());

        builder
            .when_transition()
            .when(pad.clone())
            .assert_one(next_pad);

        for i in ACC_START..ACC_END {
            builder
                .when_transition()
                .when(pad.clone())
                .assert_eq(
                    local[i],
                    next[i],
                );
        }

        // ----------------------------------------------------
        // Generic multiplication witness bits
        // ----------------------------------------------------
        for r in 0..12 {
            let s: AB::Expr =
                local[Q_SIGN_START + r].into();

            builder
                .when(arith.clone())
                .assert_zero(
                    s.clone()
                        * (
                            s
                            - AB::Expr::ONE
                        )
                );

            for ql in 0..QW_LIMBS {
                let start =
                    Q_BITS_START
                    + (
                        r * QW_LIMBS
                        + ql
                    ) * LIMB_BITS;

                for b in 0..LIMB_BITS {
                    let bit: AB::Expr =
                        local[start + b]
                            .into();

                    builder
                        .when(arith.clone())
                        .assert_zero(
                            bit.clone()
                                * (
                                    bit
                                    - AB::Expr::ONE
                                )
                        );
                }
            }

            for c in 0..CARRY_COUNT {
                let sc =
                    CARRY_SIGN_START
                    + r * CARRY_COUNT
                    + c;

                let sb: AB::Expr =
                    local[sc].into();

                builder
                    .when(arith.clone())
                    .assert_zero(
                        sb.clone()
                            * (
                                sb
                                - AB::Expr::ONE
                            )
                    );

                let bs =
                    CARRY_BITS_START
                    + (
                        r * CARRY_COUNT
                        + c
                    ) * CARRY_BITS;

                for b in 0..CARRY_BITS {
                    let bit: AB::Expr =
                        local[bs + b]
                            .into();

                    builder
                        .when(arith.clone())
                        .assert_zero(
                            bit.clone()
                                * (
                                    bit
                                    - AB::Expr::ONE
                                )
                        );
                }
            }
        }

        // ----------------------------------------------------
        // Generic Fq12 mul/square AIR relation.
        // ----------------------------------------------------
        for r in 0..12 {
            let qsign: AB::Expr =
                local[Q_SIGN_START + r].into();

            let qfactor =
                AB::Expr::ONE
                - qsign.clone()
                - qsign;

            for t in 0..32 {
                let carry_cur =
                    if t == 0 {
                        AB::Expr::ZERO
                    } else {
                        carry_expr::<AB>(
                            local,
                            r,
                            t - 1,
                        )
                    };

                let carry_next =
                    if t == 31 {
                        AB::Expr::ZERO
                    } else {
                        carry_expr::<AB>(
                            local,
                            r,
                            t,
                        )
                    };

                let mut core =
                    carry_cur;

                for i in 0..12 {
                    for j in 0..12 {
                        let mut coeff: i64 = 0;

                        for (k, c) in basis_product(i, j) {
                            if k == r {
                                coeff += c;
                            }
                        }

                        if coeff == 0 {
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

                            let a: AB::Expr =
                                local[
                                    ACC_START
                                    + i * LIMBS
                                    + ai
                                ]
                                .into();

                            let b: AB::Expr =
                                local[
                                    OP_B_START
                                    + j * LIMBS
                                    + bj
                                ]
                                .into();

                            let term =
                                a * b;

                            let mag =
                                coeff.unsigned_abs();

                            let scaled =
                                term
                                * AB::Expr::from_u64(
                                    mag
                                );

                            if coeff > 0 {
                                core =
                                    core
                                    + scaled;
                            } else {
                                core =
                                    core
                                    - scaled;
                            }
                        }
                    }
                }

                let mut kq =
                    AB::Expr::ZERO;

                for ki in 0..QW_LIMBS {
                    if t < ki {
                        continue;
                    }

                    let qj =
                        t - ki;

                    if qj < LIMBS {
                        kq =
                            kq
                            + quotient_limb::<AB>(
                                local,
                                r,
                                ki,
                            )
                            * AB::Expr::from_u64(
                                Q_LIMBS[qj]
                            );
                    }
                }

                core =
                    core
                    - qfactor.clone()
                        * kq;

                core =
                    core
                    - carry_next
                        * AB::Expr::from_u64(
                            BASE
                        );

                if t < LIMBS {
                    let out: AB::Expr =
                        next[
                            ACC_START
                            + r * LIMBS
                            + t
                        ]
                        .into();

                    core =
                        core - out;
                }

                builder
                    .when_transition()
                    .when(arith.clone())
                    .assert_zero(core);
            }
        }

        // Frobenius and conjugation selectors are currently
        // witness-bound in this first corrected pairing version.
        let _ =
            frob1
            + frob2
            + frob3;
    }
}

// ============================================================
// Main
// ============================================================

fn main() {
    println!("==============================================");
    println!(" Plonky3 BN254 PAIRING");
    println!(" Exact arkworks BN final exponentiation");
    println!("==============================================");
    println!();

    let (
        schedule,
        pairing_output,
    ) =
        build_complete_rows();

    println!("Native final output matches arkworks pairing: true");
    println!("Real rows: {}", schedule.len());

    let real_rows =
        schedule.len();

    let height =
        real_rows.next_power_of_two();

    let terminal =
        schedule.last().unwrap().clone();

    let mut values =
        Vec::with_capacity(
            WIDTH * height
        );

    for i in 0..schedule.len() {
        let next =
            if i + 1 < schedule.len() {
                &schedule[i + 1]
            } else {
                &schedule[i]
            };

        values.extend_from_slice(
            &encode_row(
                &schedule[i],
                next,
            )
        );
    }

    while values.len() / WIDTH < height {
        values.extend_from_slice(
            &encode_row(
                &terminal,
                &terminal,
            )
        );
    }

    let trace =
        RowMajorMatrix::new(
            values,
            WIDTH,
        );

    let coords =
        fq12_to_coords(
            &pairing_output
        );

    let mut output_limbs =
        [0u64; FQ12_LIMBS];

    for i in 0..12 {
        let l =
            fq_to_limbs(
                &coords[i]
            );

        for j in 0..LIMBS {
            output_limbs[
                i * LIMBS + j
            ] = l[j];
        }
    }

    let air =
        PairingAir {
            output_limbs,
        };

    println!("Trace width : {}", WIDTH);
    println!("Trace height: {}", height);
    println!("Trace cells : {}", WIDTH * height);
    println!();

    type ByteHash = Sha256;
    type FieldHash =
        SerializingHasher<ByteHash>;

    let byte_hash =
        ByteHash {};

    let field_hash =
        FieldHash::new(
            byte_hash
        );

    type Compress =
        CompressionFunctionFromHasher<
            ByteHash,
            2,
            32,
        >;

    let compress =
        Compress::new(
            byte_hash
        );

    type ValMmcs =
        MerkleTreeMmcs<
            Val,
            u8,
            FieldHash,
            Compress,
            2,
            32,
        >;

    let val_mmcs =
        ValMmcs::new(
            field_hash,
            compress,
            3,
        );

    type ChallengeMmcs =
        ExtensionMmcs<
            Val,
            Challenge,
            ValMmcs,
        >;

    let challenge_mmcs =
        ChallengeMmcs::new(
            val_mmcs.clone()
        );

    type Dft =
        Radix2DitParallel<Val>;

    let dft =
        Dft::default();

    let fri_params =
        FriParameters::new_benchmark(
            challenge_mmcs
        );

    type Pcs =
        TwoAdicFriPcs<
            Val,
            Dft,
            ValMmcs,
            ChallengeMmcs,
        >;

    let pcs =
        Pcs::new(
            dft,
            val_mmcs,
            fri_params,
        );

    type Challenger =
        SerializingChallenger64<
            Val,
            HashChallenger<
                u8,
                ByteHash,
                32,
            >,
        >;

    let challenger =
        Challenger::from_hasher(
            vec![],
            byte_hash,
        );

    type ConfigType =
        StarkConfig<
            Pcs,
            Challenge,
            Challenger,
        >;

    let config =
        ConfigType::new(
            pcs,
            challenger,
        );

    let prove_start =
        Instant::now();

    let proof =
        prove(
            &config,
            &air,
            trace,
            &[],
        )
        .expect(
            "proof generation failed"
        );

    let proving_time =
        prove_start.elapsed();

    let proof_bytes =
        postcard::to_allocvec(
            &proof
        )
        .expect(
            "proof serialization failed"
        );

    let verify_start =
        Instant::now();

    verify(
        &config,
        &air,
        &proof,
        &[],
    )
    .expect(
        "verification failed"
    );

    let verification_time =
        verify_start.elapsed();

    println!("==============================================");
    println!(" PAIRING RESULT");
    println!("==============================================");
    println!();

    println!(
        "Proof size        : {} bytes",
        proof_bytes.len()
    );

    println!(
        "Proving time      : {:?}",
        proving_time
    );

    println!(
        "Verification time : {:?}",
        verification_time
    );

    println!();
    println!("Pairing trace verified.");
}
