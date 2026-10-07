
use std::time::Instant;

use ark_bn254::{Bn254, Config, Fq, Fq2, Fq6, Fq12, Fr, G1Affine, G2Affine};
use ark_ec::{
    bn::{BnConfig, G2Prepared},
    pairing::Pairing,
    AffineRepr,
    CurveGroup,
};
use ark_ff::{BigInteger, Field, One, PrimeField, Zero};
use num_bigint::{BigInt, BigUint, Sign};

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_air::symbolic::{
    AirLayout,
    get_max_constraint_degree,
    get_symbolic_constraints,
};
use p3_challenger::{HashChallenger, SerializingChallenger64};
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::{extension::BinomialExtensionField, PrimeCharacteristicRing};
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_goldilocks::Goldilocks;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_sha256::Sha256;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use p3_uni_stark::{prove, verify, StarkConfig};

type Val = Goldilocks;
type Challenge = BinomialExtensionField<Val, 2>;

const LIMBS: usize = 16;
const LIMB_BITS: usize = 16;

// Final monolithic public statement:
//
//   C_f1, C_F1, C_f2, C_F2, C_f3, C_F3, W
//
// D_1,D_2,D_3 are NOT public.  They live only in the difference subtrace.
const PUBLIC_G1_POINTS: usize = 7;
const COMMITMENT_POINTS: usize = 6;
const G1_POINT_LIMBS: usize = 2 * LIMBS;
const NUM_PUBLIC_VALUES: usize = PUBLIC_G1_POINTS * G1_POINT_LIMBS;
const BASE: u64 = 1 << LIMB_BITS;

const FQ12_COORDS: usize = 12;
const FQ12_LIMBS: usize = FQ12_COORDS * LIMBS;

const Q_LIMBS: [u64; LIMBS] = [
    64839, 55420, 35862, 15392,
    51853, 26737, 27281, 38785,
    22621, 33153, 17846, 47184,
    41001, 57649, 20082, 12388,
];

const MILLER_STEPS: usize = 411;

// Exact exponent implemented by arkworks' BN optimal-Ate final map.
// It has 2980 bits for BN254.
const EXP_BITS: usize = 2980;
const EXP_LIMBS: usize = (EXP_BITS + LIMB_BITS - 1) / LIMB_BITS;

// Signed quotient/carry witness sizing.
const QW_LIMBS: usize = 17;
const CARRY_COUNT: usize = 31;
const CARRY_BITS: usize = 48;

// ============================================================
// Global operation selectors
// ============================================================

const SEL_MILLER_SQ: usize = 0;
const SEL_MILLER_LINE: usize = 1;
const SEL_EXP_INIT: usize = 2;
const SEL_EXP_ACC: usize = 3;
const SEL_EXP_BASE: usize = 4;
const SEL_OUTPUT: usize = 5;
const SEL_PAD: usize = 6;
const NUM_SELECTORS: usize = 7;

// ============================================================
// Metadata
// ============================================================

const COEFF_INDEX: usize = NUM_SELECTORS;
const MILLER_STEP_INDEX: usize = COEFF_INDEX + 1;
const EXP_COUNTER: usize = MILLER_STEP_INDEX + 1;

// ============================================================
// State registers
// ============================================================

const ACC_START: usize = EXP_COUNTER + 1;
const ACC_END: usize = ACC_START + FQ12_LIMBS;

const BASE_START: usize = ACC_END;
const BASE_END: usize = BASE_START + FQ12_LIMBS;

const EXP_START: usize = BASE_END;
const EXP_END: usize = EXP_START + EXP_LIMBS;

// Evaluated Miller sparse line c0,c3,c4 = 6 Fq = 96 limbs.
const LINE_LIMBS: usize = 6 * LIMBS;
const LINE_START: usize = EXP_END;
const LINE_END: usize = LINE_START + LINE_LIMBS;

// One-hot identity for the exact Miller operation schedule.
const MILLER_ID_START: usize = LINE_END;
const MILLER_ID_END: usize = MILLER_ID_START + MILLER_STEPS;

// State range bits.
const ACC_BITS_START: usize = MILLER_ID_END;
const ACC_BITS_END: usize = ACC_BITS_START + FQ12_LIMBS * LIMB_BITS;

const BASE_BITS_START: usize = ACC_BITS_END;
const BASE_BITS_END: usize = BASE_BITS_START + FQ12_LIMBS * LIMB_BITS;

const EXP_BITS_START: usize = BASE_BITS_END;
const EXP_RANGE_BITS_END: usize = EXP_BITS_START + EXP_LIMBS * LIMB_BITS;

// Generic Fq12 multiplication operands A,B.
const OP_A_START: usize = EXP_RANGE_BITS_END;
const OP_A_END: usize = OP_A_START + FQ12_LIMBS;

const OP_B_START: usize = OP_A_END;
const OP_B_END: usize = OP_B_START + FQ12_LIMBS;

// Modular-reduction witness for 12 Fq outputs.
const Q_SIGN_START: usize = OP_B_END;
const Q_SIGN_END: usize = Q_SIGN_START + FQ12_COORDS;

const Q_BITS_START: usize = Q_SIGN_END;
const Q_BITS_END: usize =
    Q_BITS_START + FQ12_COORDS * QW_LIMBS * LIMB_BITS;

const CARRY_SIGN_START: usize = Q_BITS_END;
const CARRY_SIGN_END: usize =
    CARRY_SIGN_START + FQ12_COORDS * CARRY_COUNT;

const CARRY_BITS_START: usize = CARRY_SIGN_END;
const PAIR_WIDTH: usize =
    CARRY_BITS_START + FQ12_COORDS * CARRY_COUNT * CARRY_BITS;

// ============================================================
// Native row / Miller constants
// ============================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    MillerSquare,
    MillerLine,
    ExpInit,
    ExpAcc,
    ExpBase,
    Output,
    Pad,
}

#[derive(Clone)]
struct NativeRow {
    op: Op,
    coeff_index: usize,
    miller_step: Option<usize>,
    exp_counter: usize,
    acc: Fq12,
    base: Fq12,
    exp: BigUint,
    line: Option<(Fq2, Fq2, Fq2)>,
}

#[derive(Clone)]
struct MillerStepConst {
    is_square: bool,
    coeff_index: usize,

    // Which G1 input is used on this Miller-line row:
    //   0 = D_1, 1 = D_2, 2 = D_3, 3 = W.
    // Square rows use 0 and zero raw coefficients.
    pair_index: usize,

    // RAW prepared G2 line coefficients (c0,c3,c4), before multiplying
    // c0 by y_P and c3 by x_P.  The AIR dynamically performs those
    // four Fq multiplications from the internal/public G1 coordinates.
    raw_coeff_limbs: [u64; LINE_LIMBS],
}

// ============================================================
// Fq12 basis helpers
//
// index = w*6 + v*2 + u
// u^2=-1, v^3=9+u, w^2=v.
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
    // mul_by_034(c0,c3,c4) represents:
    //
    // c0.c0 = c0
    // c1.c0 = c3
    // c1.c1 = c4
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
// Big-int / limb helpers
// ============================================================

fn fq_to_biguint(x: &Fq) -> BigUint {
    BigUint::from_bytes_le(&x.into_bigint().to_bytes_le())
}

fn modulus_biguint() -> BigUint {
    BigUint::from_bytes_le(&Fq::MODULUS.to_bytes_le())
}

fn scalar_modulus_biguint() -> BigUint {
    BigUint::from_bytes_le(&Fr::MODULUS.to_bytes_le())
}

fn final_exponent_biguint() -> BigUint {
    // Derive the exact single exponent represented by arkworks'
    // BN final-exponentiation DAG.
    //
    // Easy part:
    //   R = (q^6 - 1)(q^2 + 1)
    //
    // Hard part, for positive BN parameter x:
    //   A = 12x^3 + 6x^2 + 4x
    //   B = 12x^3 + 6x^2 + 6x
    //   D = 12x^3 + 12x^2 + 6x + 1
    //
    //   H = q^3(A - 1) + q^2 B + q A + D
    //
    // Therefore:
    //   E = R * H
    //
    // and pairing = miller_output^E.
    let q = modulus_biguint();

    // BN254 arkworks Config::X is one 64-bit limb.
    assert_eq!(Config::X.len(), 1);
    assert!(!Config::X_IS_NEGATIVE);

    let x = BigUint::from(Config::X[0]);

    let x2 = &x * &x;
    let x3 = &x2 * &x;

    let a =
        BigUint::from(12u64) * &x3
        + BigUint::from(6u64) * &x2
        + BigUint::from(4u64) * &x;

    let b =
        BigUint::from(12u64) * &x3
        + BigUint::from(6u64) * &x2
        + BigUint::from(6u64) * &x;

    let d =
        BigUint::from(12u64) * &x3
        + BigUint::from(12u64) * &x2
        + BigUint::from(6u64) * &x
        + BigUint::from(1u64);

    let easy =
        (q.pow(6u32) - BigUint::from(1u64))
        * (q.pow(2u32) + BigUint::from(1u64));

    let hard =
        q.pow(3u32) * (&a - BigUint::from(1u64))
        + q.pow(2u32) * &b
        + &q * &a
        + d;

    let e = easy * hard;

    assert_eq!(e.bits() as usize, EXP_BITS);

    e
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

fn exp_to_limbs(x: &BigUint) -> [u64; EXP_LIMBS] {
    biguint_to_limbs::<EXP_LIMBS>(x)
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
// Evaluated BN254 Miller line
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

fn line_to_limbs(line: &(Fq2, Fq2, Fq2)) -> [u64; LINE_LIMBS] {
    let mut out = [0u64; LINE_LIMBS];
    let fqs = [
        line.0.c0, line.0.c1,
        line.1.c0, line.1.c1,
        line.2.c0, line.2.c1,
    ];

    for i in 0..6 {
        let limbs = fq_to_limbs(&fqs[i]);
        for j in 0..LIMBS {
            out[i * LIMBS + j] = limbs[j];
        }
    }

    out
}

fn raw_coeff_to_limbs(coeff: &(Fq2, Fq2, Fq2)) -> [u64; LINE_LIMBS] {
    line_to_limbs(coeff)
}

// ============================================================
// Build exact fixed Miller program for e(G1,G2)
// ============================================================

fn build_miller_program(
    pairs: &[(G1Affine, G2Affine); 4],
) -> (Vec<NativeRow>, Vec<MillerStepConst>, Fq12) {
    // Prepare each G2 point once.  All four prepared points use the same
    // BN254 Miller-loop digit schedule and therefore the same coefficient
    // index at each logical line stage.
    let prepared: Vec<G2Prepared<Config>> = pairs
        .iter()
        .map(|(_, q)| G2Prepared::<Config>::from(*q))
        .collect();

    let mut rows = Vec::<NativeRow>::new();
    let mut constants = Vec::<MillerStepConst>::new();

    // f accumulates the PRODUCT of all four Miller-loop contributions.
    let mut f = Fq12::one();

    let mut coeff_index = 0usize;
    let mut step = 0usize;

    for i in (1..Config::ATE_LOOP_COUNT.len()).rev() {
        let digit = Config::ATE_LOOP_COUNT[i - 1];

        // A multi-pairing shares the Miller squaring across every pair.
        if i != Config::ATE_LOOP_COUNT.len() - 1 {
            rows.push(NativeRow {
                op: Op::MillerSquare,
                coeff_index,
                miller_step: Some(step),
                exp_counter: 0,
                acc: f,
                base: Fq12::zero(),
                exp: BigUint::from(0u64),
                line: None,
            });

            constants.push(MillerStepConst {
                is_square: true,
                coeff_index,
                pair_index: 0,
                raw_coeff_limbs: [0u64; LINE_LIMBS],
            });

            let direct = fq12_mul_direct(&f, &f);
            let mut ark = f;
            ark.square_in_place();
            assert_eq!(direct, ark);

            f = ark;
            step += 1;
        }

        // For the current BN254 line coefficient, multiply the evaluated
        // line contribution from EACH pairing input into the same accumulator.
        for pair_index in 0..pairs.len() {
            let coeff = prepared[pair_index].ell_coeffs[coeff_index];
            let line = evaluated_line(&coeff, &pairs[pair_index].0);

            rows.push(NativeRow {
                op: Op::MillerLine,
                coeff_index,
                miller_step: Some(step),
                exp_counter: 0,
                acc: f,
                base: Fq12::zero(),
                exp: BigUint::from(0u64),
                line: Some(line),
            });

            constants.push(MillerStepConst {
                is_square: false,
                coeff_index,
                pair_index,
                raw_coeff_limbs: raw_coeff_to_limbs(&coeff),
            });

            let sparse = sparse_line_fq12(&line);
            let direct = fq12_mul_direct(&f, &sparse);

            f.mul_by_034(&line.0, &line.1, &line.2);

            assert_eq!(
                direct,
                f,
                "sparse direct Fq12 multiplication != arkworks mul_by_034"
            );

            step += 1;
        }

        coeff_index += 1;

        // Signed Miller digit introduces a second line coefficient.
        if digit == 1 || digit == -1 {
            for pair_index in 0..pairs.len() {
                let coeff = prepared[pair_index].ell_coeffs[coeff_index];
                let line = evaluated_line(&coeff, &pairs[pair_index].0);

                rows.push(NativeRow {
                    op: Op::MillerLine,
                    coeff_index,
                    miller_step: Some(step),
                    exp_counter: 0,
                    acc: f,
                    base: Fq12::zero(),
                    exp: BigUint::from(0u64),
                    line: Some(line),
                });

                constants.push(MillerStepConst {
                    is_square: false,
                    coeff_index,
                    pair_index,
                    raw_coeff_limbs: raw_coeff_to_limbs(&coeff),
                });

                let sparse = sparse_line_fq12(&line);
                let direct = fq12_mul_direct(&f, &sparse);

                f.mul_by_034(&line.0, &line.1, &line.2);
                assert_eq!(direct, f);

                step += 1;
            }

            coeff_index += 1;
        }
    }

    assert!(
        !Config::X_IS_NEGATIVE,
        "this complete pairing trace assumes BN254 X_IS_NEGATIVE=false"
    );

    // Final two BN254 line coefficients.
    for _ in 0..2 {
        for pair_index in 0..pairs.len() {
            let coeff = prepared[pair_index].ell_coeffs[coeff_index];
            let line = evaluated_line(&coeff, &pairs[pair_index].0);

            rows.push(NativeRow {
                op: Op::MillerLine,
                coeff_index,
                miller_step: Some(step),
                exp_counter: 0,
                acc: f,
                base: Fq12::zero(),
                exp: BigUint::from(0u64),
                line: Some(line),
            });

            constants.push(MillerStepConst {
                is_square: false,
                coeff_index,
                pair_index,
                raw_coeff_limbs: raw_coeff_to_limbs(&coeff),
            });

            let sparse = sparse_line_fq12(&line);
            let direct = fq12_mul_direct(&f, &sparse);

            f.mul_by_034(&line.0, &line.1, &line.2);
            assert_eq!(direct, f);

            step += 1;
        }

        coeff_index += 1;
    }

    for prep in &prepared {
        assert_eq!(coeff_index, prep.ell_coeffs.len());
    }

    assert_eq!(step, MILLER_STEPS);
    assert_eq!(constants.len(), MILLER_STEPS);

    (rows, constants, f)
}

// ============================================================
// Full pairing schedule:
//
// Miller
// -> ExpInit
// -> 2790 x (ExpAcc, ExpBase)
// -> Output
// -> Pad
// ============================================================

fn build_full_schedule(
    pairs: &[(G1Affine, G2Affine); 4],
) -> (Vec<NativeRow>, Vec<MillerStepConst>, Fq12) {
    let (mut rows, miller_constants, miller_out) =
        build_miller_program(pairs);

    let final_exp = final_exponent_biguint();
    assert_eq!(final_exp.bits() as usize, EXP_BITS);

    // Transition row: Miller product -> one shared final exponentiation.
    rows.push(NativeRow {
        op: Op::ExpInit,
        coeff_index: 87,
        miller_step: None,
        exp_counter: 0,
        acc: miller_out,
        base: Fq12::zero(),
        exp: BigUint::from(0u64),
        line: None,
    });

    let mut acc = Fq12::one();
    let mut base = miller_out;
    let mut exp = final_exp.clone();

    for counter in 0..EXP_BITS {
        rows.push(NativeRow {
            op: Op::ExpAcc,
            coeff_index: 87,
            miller_step: None,
            exp_counter: counter,
            acc,
            base,
            exp: exp.clone(),
            line: None,
        });

        if (&exp & BigUint::from(1u64)) == BigUint::from(1u64) {
            acc = fq12_mul_direct(&acc, &base);
        }

        rows.push(NativeRow {
            op: Op::ExpBase,
            coeff_index: 87,
            miller_step: None,
            exp_counter: counter,
            acc,
            base,
            exp: exp.clone(),
            line: None,
        });

        base = fq12_mul_direct(&base, &base);
        exp >>= 1usize;
    }

    assert!(exp.is_zero());

    let ps: Vec<G1Affine> = pairs.iter().map(|(p, _)| *p).collect();
    let qs: Vec<G2Affine> = pairs.iter().map(|(_, q)| *q).collect();

    let direct_product = Bn254::multi_pairing(ps, qs).0;

    assert_eq!(
        acc,
        direct_product,
        "binary final exponentiation result != arkworks multi_pairing"
    );

    rows.push(NativeRow {
        op: Op::Output,
        coeff_index: 87,
        miller_step: None,
        exp_counter: EXP_BITS,
        acc,
        base,
        exp: BigUint::from(0u64),
        line: None,
    });

    rows.push(NativeRow {
        op: Op::Pad,
        coeff_index: 87,
        miller_step: None,
        exp_counter: EXP_BITS,
        acc,
        base,
        exp: BigUint::from(0u64),
        line: None,
    });

    (rows, miller_constants, direct_product)
}

// ============================================================
// Generic Fq12 multiplication witness
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
            &s - BigInt::from_biguint(Sign::Plus, out_big.clone());

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

                assert!(
                    mag < (1u128 << CARRY_BITS),
                    "carry does not fit"
                );

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
// Encoding helpers
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

fn push_exp(row: &mut Vec<Val>, x: &BigUint) {
    for limb in exp_to_limbs(x) {
        row.push(Val::from_u64(limb));
    }
}

fn push_selectors(row: &mut Vec<Val>, op: Op) {
    let active = match op {
        Op::MillerSquare => SEL_MILLER_SQ,
        Op::MillerLine => SEL_MILLER_LINE,
        Op::ExpInit => SEL_EXP_INIT,
        Op::ExpAcc => SEL_EXP_ACC,
        Op::ExpBase => SEL_EXP_BASE,
        Op::Output => SEL_OUTPUT,
        Op::Pad => SEL_PAD,
    };

    for i in 0..NUM_SELECTORS {
        row.push(if i == active { Val::ONE } else { Val::ZERO });
    }
}

fn arithmetic_operands(
    row: &NativeRow,
    next: &NativeRow,
) -> Option<(Fq12, Fq12, Fq12)> {
    match row.op {
        Op::MillerSquare => Some((row.acc, row.acc, next.acc)),

        Op::MillerLine => {
            let sparse = sparse_line_fq12(
                row.line.as_ref().unwrap()
            );
            Some((row.acc, sparse, next.acc))
        }

        Op::ExpAcc => {
            let bit =
                (&row.exp & BigUint::from(1u64))
                == BigUint::from(1u64);

            if bit {
                Some((row.acc, row.base, next.acc))
            } else {
                None
            }
        }

        Op::ExpBase => Some((row.base, row.base, next.base)),

        _ => None,
    }
}

fn encode_row(
    native: &NativeRow,
    next: &NativeRow,
) -> Vec<Val> {
    let mut row = Vec::with_capacity(PAIR_WIDTH);

    push_selectors(&mut row, native.op);

    row.push(Val::from_u64(native.coeff_index as u64));
    row.push(Val::from_u64(
        native.miller_step.unwrap_or(0) as u64
    ));
    row.push(Val::from_u64(native.exp_counter as u64));

    push_fq12(&mut row, &native.acc);
    push_fq12(&mut row, &native.base);
    push_exp(&mut row, &native.exp);

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

    for i in 0..MILLER_STEPS {
        row.push(
            if native.miller_step == Some(i) {
                Val::ONE
            } else {
                Val::ZERO
            }
        );
    }

    // State range bits.
    for fq in fq12_to_coords(&native.acc) {
        for limb in fq_to_limbs(&fq) {
            push_bits_u64(&mut row, limb, LIMB_BITS);
        }
    }

    for fq in fq12_to_coords(&native.base) {
        for limb in fq_to_limbs(&fq) {
            push_bits_u64(&mut row, limb, LIMB_BITS);
        }
    }

    for limb in exp_to_limbs(&native.exp) {
        push_bits_u64(&mut row, limb, LIMB_BITS);
    }

    if let Some((a, b, out)) = arithmetic_operands(native, next) {
        push_fq12(&mut row, &a);
        push_fq12(&mut row, &b);

        let w = build_mul_witness(&a, &b, &out);

        for r in 0..12 {
            row.push(if w.q_signs[r] { Val::ONE } else { Val::ZERO });
        }

        for r in 0..12 {
            for l in 0..QW_LIMBS {
                push_bits_u64(&mut row, w.q_limbs[r][l], LIMB_BITS);
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
        for _ in 0..(2 * FQ12_LIMBS) {
            row.push(Val::ZERO);
        }
        for _ in Q_SIGN_START..PAIR_WIDTH {
            row.push(Val::ZERO);
        }
    }

    assert_eq!(row.len(), PAIR_WIDTH);
    row
}

// ============================================================
// Trace construction
// ============================================================

fn make_trace(
    pairs: &[(G1Affine, G2Affine); 4],
    commitment_points: &[G1Affine; COMMITMENT_POINTS],
    d_points: &[G1Affine; 3],
) -> (
    RowMajorMatrix<Val>,
    usize,
    usize,
    Vec<MillerStepConst>,
    Fq12,
) {
    let (schedule, constants, pairing_product) =
        build_full_schedule(pairs);

    let real_rows = schedule.len();
    let height = real_rows.next_power_of_two();

    // First build the pairing subtrace.
    let mut pair_values = Vec::with_capacity(PAIR_WIDTH * height);

    for i in 0..schedule.len() {
        let next =
            if i + 1 < schedule.len() {
                &schedule[i + 1]
            } else {
                &schedule[i]
            };

        pair_values.extend_from_slice(
            &encode_row(&schedule[i], next)
        );
    }

    let terminal = schedule.last().unwrap().clone();

    while pair_values.len() / PAIR_WIDTH < height {
        pair_values.extend_from_slice(
            &encode_row(&terminal, &terminal)
        );
    }

    // Build the dynamic Miller-line evaluation witness at the SAME height.
    let line_values =
        line_eval_layer::make_values_with_height(
            &schedule,
            &constants,
            pairs,
            height,
        );

    // Build the commitment-difference subtrace at the SAME padded height.
    let diff_values =
        diff_layer::make_values_with_height(
            commitment_points,
            d_points,
            height,
        );

    assert_eq!(pair_values.len(), PAIR_WIDTH * height);
    assert_eq!(line_values.len(), line_eval_layer::WIDTH * height);
    assert_eq!(diff_values.len(), diff_layer::WIDTH * height);

    let combined_width = PAIR_WIDTH + line_eval_layer::WIDTH + diff_layer::WIDTH;
    let mut values = Vec::with_capacity(combined_width * height);

    for r in 0..height {
        let ps = r * PAIR_WIDTH;
        let ls = r * line_eval_layer::WIDTH;
        let ds = r * diff_layer::WIDTH;

        values.extend_from_slice(&pair_values[ps..ps + PAIR_WIDTH]);
        values.extend_from_slice(&line_values[ls..ls + line_eval_layer::WIDTH]);
        values.extend_from_slice(&diff_values[ds..ds + diff_layer::WIDTH]);
    }

    (
        RowMajorMatrix::new(values, combined_width),
        real_rows,
        height,
        constants,
        pairing_product,
    )
}

// ============================================================
// AIR expression helpers
// ============================================================

fn expr_from_bits<AB: AirBuilder>(
    row: &[AB::Var],
    start: usize,
    bit_count: usize,
) -> AB::Expr {
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

    (AB::Expr::ONE - sign.clone() - sign) * mag
}

fn state_limb<AB: AirBuilder>(
    row: &[AB::Var],
    start: usize,
    coord: usize,
    limb: usize,
) -> AB::Expr {
    row[start + coord * LIMBS + limb].into()
}

fn operand_limb<AB: AirBuilder>(
    row: &[AB::Var],
    start: usize,
    coord: usize,
    limb: usize,
) -> AB::Expr {
    row[start + coord * LIMBS + limb].into()
}

fn quotient_limb<AB: AirBuilder>(
    row: &[AB::Var],
    r: usize,
    l: usize,
) -> AB::Expr {
    let start =
        Q_BITS_START + (r * QW_LIMBS + l) * LIMB_BITS;
    expr_from_bits::<AB>(row, start, LIMB_BITS)
}

fn carry_expr<AB: AirBuilder>(
    row: &[AB::Var],
    r: usize,
    c: usize,
) -> AB::Expr {
    let sign_col =
        CARRY_SIGN_START + r * CARRY_COUNT + c;

    let bits_start =
        CARRY_BITS_START
        + (r * CARRY_COUNT + c) * CARRY_BITS;

    signed_expr_from_bits::<AB>(
        row,
        sign_col,
        bits_start,
        CARRY_BITS,
    )
}

mod diff_layer {
    use super::*;

    const LIMBS: usize = 16;
    const LIMB_BITS: usize = 16;
    const BASE: u64 = 1 << LIMB_BITS;

    const Q_LIMBS: [u64; LIMBS] = [
        64839, 55420, 35862, 15392,
        51853, 26737, 27281, 38785,
        22621, 33153, 17846, 47184,
        41001, 57649, 20082, 12388,
    ];

    const NUM_RELATIONS: usize = 3;
    const NUM_PUBLIC_POINTS: usize = 6;
    const POINT_LIMBS: usize = 2 * LIMBS;
    const NUM_PUBLIC_VALUES: usize = NUM_PUBLIC_POINTS * POINT_LIMBS;

    const GLOBAL_ELEMS: usize = 36;
    const ELEM_BITS: usize = LIMBS * LIMB_BITS;
    const ELEM_WIDTH: usize = LIMBS + ELEM_BITS;
    const GLOBAL_START: usize = 0;
    const GLOBAL_END: usize = GLOBAL_START + GLOBAL_ELEMS * ELEM_WIDTH;

    const NUM_LINEAR_EQS: usize = 15;
    const LIN_K_BITS: usize = 2;
    const LIN_CARRY_BITS: usize = 5;
    const LIN_CARRY_OFFSET: i64 = 8;
    const LIN_EQ_WIDTH: usize = LIN_K_BITS + LIMBS * LIN_CARRY_BITS;
    const LIN_START: usize = GLOBAL_END;
    const LIN_END: usize = LIN_START + NUM_LINEAR_EQS * LIN_EQ_WIDTH;

    const NUM_MUL_EQS: usize = 9;
    const CARRY_COUNT: usize = 31;
    const MUL_CARRY_BITS: usize = 48;

    const SEL_START: usize = LIN_END;
    const SEL_END: usize = SEL_START + NUM_MUL_EQS;
    const A_START: usize = SEL_END;
    const B_START: usize = A_START + LIMBS;
    const C_START: usize = B_START + LIMBS;
    const A_BITS_START: usize = C_START + LIMBS;
    const B_BITS_START: usize = A_BITS_START + LIMBS * LIMB_BITS;
    const C_BITS_START: usize = B_BITS_START + LIMBS * LIMB_BITS;
    const K_BITS_START: usize = C_BITS_START + LIMBS * LIMB_BITS;
    const MUL_CARRY_SIGN_START: usize = K_BITS_START + LIMBS * LIMB_BITS;
    const MUL_CARRY_BITS_START: usize = MUL_CARRY_SIGN_START + CARRY_COUNT;
    pub(super) const WIDTH: usize = MUL_CARRY_BITS_START + CARRY_COUNT * MUL_CARRY_BITS;

    const PUB_ELEM_COUNT: usize = 12;
    const D_ELEM_BASE: usize = 12;
    const INTERMEDIATE_BASE: usize = 18;

    const I_LAMBDA: usize = 0;
    const I_DX: usize = 1;
    const I_DY: usize = 2;
    const I_RHSX: usize = 3;
    const I_XDIFF: usize = 4;
    const I_RHSY: usize = 5;

    #[derive(Clone)]
    struct MulEquation {
        a: Fq,
        b: Fq,
        c: Fq,
    }

    #[derive(Clone)]
    struct MulWitness {
        k_limbs: [u64; LIMBS],
        carry_signs: [bool; CARRY_COUNT],
        carry_mags: [u64; CARRY_COUNT],
    }

    #[derive(Clone)]
    struct LinWitness {
        k: u64,
        carries_shifted: [u64; LIMBS],
    }

    #[derive(Clone)]
    struct InstanceData {
        elems: [[u64; LIMBS]; GLOBAL_ELEMS],
        mul_eqs: [MulEquation; NUM_MUL_EQS],
        lin_witnesses: [LinWitness; NUM_LINEAR_EQS],
    }



    fn pub_x_elem(point: usize) -> usize { 2 * point }
    fn pub_y_elem(point: usize) -> usize { 2 * point + 1 }
    pub(super) fn d_x_elem(i: usize) -> usize { D_ELEM_BASE + 2 * i }
    pub(super) fn d_y_elem(i: usize) -> usize { D_ELEM_BASE + 2 * i + 1 }
    fn inter_elem(i: usize, which: usize) -> usize { INTERMEDIATE_BASE + 6 * i + which }

    pub(super) fn elem_limb_col(elem: usize, limb: usize) -> usize {
        GLOBAL_START + elem * ELEM_WIDTH + limb
    }
    fn elem_bits_col(elem: usize, limb: usize, bit: usize) -> usize {
        GLOBAL_START + elem * ELEM_WIDTH + LIMBS + limb * LIMB_BITS + bit
    }
    fn lin_k_bit_col(eq: usize, bit: usize) -> usize {
        LIN_START + eq * LIN_EQ_WIDTH + bit
    }
    fn lin_carry_bit_col(eq: usize, limb: usize, bit: usize) -> usize {
        LIN_START + eq * LIN_EQ_WIDTH + LIN_K_BITS + limb * LIN_CARRY_BITS + bit
    }

    fn fq_to_biguint(x: &Fq) -> BigUint {
        BigUint::from_bytes_le(&x.into_bigint().to_bytes_le())
    }

    fn biguint_to_limbs<const N: usize>(x: &BigUint) -> [u64; N] {
        let mut out = [0u64; N];
        let mut t = x.clone();
        let mask = BigUint::from(BASE - 1);
        for i in 0..N {
            let limb = (&t & &mask).to_u64_digits();
            out[i] = limb.first().copied().unwrap_or(0);
            t >>= LIMB_BITS;
        }
        out
    }

    fn fq_to_limbs(x: &Fq) -> [u64; LIMBS] {
        biguint_to_limbs::<LIMBS>(&fq_to_biguint(x))
    }

    fn modulus_biguint() -> BigUint {
        let mut acc = BigUint::from(0u64);
        for i in (0..LIMBS).rev() {
            acc <<= LIMB_BITS;
            acc += Q_LIMBS[i];
        }
        acc
    }

    fn push_bits_u64(row: &mut [Val], start: usize, value: u64, bits: usize) {
        for b in 0..bits {
            row[start + b] = Val::from_u64((value >> b) & 1);
        }
    }

    fn build_mul_witness(eq: &MulEquation) -> MulWitness {
        let a_big = fq_to_biguint(&eq.a);
        let b_big = fq_to_biguint(&eq.b);
        let c_big = fq_to_biguint(&eq.c);
        let q_big = modulus_biguint();

        let prod = &a_big * &b_big;
        assert!(prod >= c_big);
        let diff = &prod - &c_big;
        assert_eq!(&diff % &q_big, BigUint::from(0u64));

        let k_big = &diff / &q_big;
        let k_limbs = biguint_to_limbs::<LIMBS>(&k_big);

        let a_limbs = fq_to_limbs(&eq.a);
        let b_limbs = fq_to_limbs(&eq.b);
        let c_limbs = fq_to_limbs(&eq.c);

        let mut carry_signs = [false; CARRY_COUNT];
        let mut carry_mags = [0u64; CARRY_COUNT];
        let mut carry: i128 = 0;

        for t in 0..32 {
            let mut x = carry;

            for i in 0..LIMBS {
                if t >= i {
                    let j = t - i;
                    if j < LIMBS {
                        x += a_limbs[i] as i128 * b_limbs[j] as i128;
                    }
                }
            }

            for i in 0..LIMBS {
                if t >= i {
                    let j = t - i;
                    if j < LIMBS {
                        x -= k_limbs[i] as i128 * Q_LIMBS[j] as i128;
                    }
                }
            }

            if t < LIMBS {
                x -= c_limbs[t] as i128;
            }

            assert_eq!(x % BASE as i128, 0);
            let next = x / BASE as i128;

            if t < CARRY_COUNT {
                let neg = next < 0;
                let mag = next.unsigned_abs();
                assert!(mag < (1u128 << MUL_CARRY_BITS));
                carry_signs[t] = neg;
                carry_mags[t] = mag as u64;
            } else {
                assert_eq!(next, 0);
            }

            carry = next;
        }

        MulWitness { k_limbs, carry_signs, carry_mags }
    }

    fn lin_witness(
        terms: &[(i64, [u64; LIMBS])],
        q_coeff: i64,
    ) -> LinWitness {
        let q_big = BigInt::from_biguint(Sign::Plus, modulus_biguint());
        let mut sum = BigInt::from(0i64);

        for (coeff, limbs) in terms {
            let mut n = BigUint::from(0u64);
            for i in (0..LIMBS).rev() {
                n <<= LIMB_BITS;
                n += limbs[i];
            }
            sum += BigInt::from(*coeff) * BigInt::from_biguint(Sign::Plus, n);
        }

        let denom = BigInt::from(q_coeff) * &q_big;
        assert!(q_coeff == 1 || q_coeff == -1);

        let k_big = -&sum / &denom;
        assert_eq!(sum + &denom * &k_big, BigInt::from(0i64));
        let k = k_big.to_biguint().unwrap().to_u64_digits().first().copied().unwrap_or(0);
        assert!(k < 4);

        let mut carries_shifted = [0u64; LIMBS];
        let mut carry: i64 = 0;

        for t in 0..LIMBS {
            let mut x = carry;
            for (coeff, limbs) in terms {
                x += *coeff * limbs[t] as i64;
            }
            x += q_coeff * k as i64 * Q_LIMBS[t] as i64;

            assert_eq!(x % BASE as i64, 0, "linear carry not integral at limb {t}");
            let next = x / BASE as i64;
            let shifted = next + LIN_CARRY_OFFSET;
            assert!(shifted >= 0 && shifted < (1 << LIN_CARRY_BITS));
            carries_shifted[t] = shifted as u64;
            carry = next;
        }
        assert_eq!(carry, 0);

        LinWitness { k, carries_shifted }
    }

    fn addition_data(
        p: G1Affine,
        q: G1Affine,
        r: G1Affine,
    ) -> ([Fq; 6], [MulEquation; 3]) {
        assert!(!p.infinity && !q.infinity && !r.infinity);
        let dx = q.x - p.x;
        assert!(!dx.is_zero());
        let dy = q.y - p.y;
        let lambda = dy * dx.inverse().unwrap();
        let rhs_x = r.x + p.x + q.x;
        let xdiff = p.x - r.x;
        let rhs_y = r.y + p.y;

        assert_eq!(lambda * dx, dy);
        assert_eq!(lambda * lambda, rhs_x);
        assert_eq!(lambda * xdiff, rhs_y);
        assert_eq!((p.into_group() + q.into_group()).into_affine(), r);

        (
            [lambda, dx, dy, rhs_x, xdiff, rhs_y],
            [
                MulEquation { a: lambda, b: dx, c: dy },
                MulEquation { a: lambda, b: lambda, c: rhs_x },
                MulEquation { a: lambda, b: xdiff, c: rhs_y },
            ],
        )
    }

    fn make_instance(
        public_points: &[G1Affine; NUM_PUBLIC_POINTS],
        d_points: &[G1Affine; NUM_RELATIONS],
    ) -> InstanceData {
        let mut elems = [[0u64; LIMBS]; GLOBAL_ELEMS];

        for p in 0..NUM_PUBLIC_POINTS {
            elems[pub_x_elem(p)] = fq_to_limbs(&public_points[p].x);
            elems[pub_y_elem(p)] = fq_to_limbs(&public_points[p].y);
        }

        for i in 0..NUM_RELATIONS {
            elems[d_x_elem(i)] = fq_to_limbs(&d_points[i].x);
            elems[d_y_elem(i)] = fq_to_limbs(&d_points[i].y);
        }

        let mut muls = Vec::with_capacity(NUM_MUL_EQS);
        let mut lin_w = Vec::with_capacity(NUM_LINEAR_EQS);

        for i in 0..NUM_RELATIONS {
            let cf = public_points[2 * i];
            let co = public_points[2 * i + 1];
            let d = d_points[i];

            let (ints, meqs) = addition_data(d, co, cf);

            for j in 0..6 {
                elems[inter_elem(i, j)] = fq_to_limbs(&ints[j]);
            }
            muls.extend_from_slice(&meqs);

            let xp = elems[d_x_elem(i)];
            let yp = elems[d_y_elem(i)];
            let xq = elems[pub_x_elem(2 * i + 1)];
            let yq = elems[pub_y_elem(2 * i + 1)];
            let xr = elems[pub_x_elem(2 * i)];
            let yr = elems[pub_y_elem(2 * i)];
            let dx = elems[inter_elem(i, I_DX)];
            let dy = elems[inter_elem(i, I_DY)];
            let rhsx = elems[inter_elem(i, I_RHSX)];
            let xdiff = elems[inter_elem(i, I_XDIFF)];
            let rhsy = elems[inter_elem(i, I_RHSY)];

            lin_w.push(lin_witness(&[(1, xq), (-1, xp), (-1, dx)], 1));
            lin_w.push(lin_witness(&[(1, yq), (-1, yp), (-1, dy)], 1));
            lin_w.push(lin_witness(&[(1, xr), (1, xp), (1, xq), (-1, rhsx)], -1));
            lin_w.push(lin_witness(&[(1, xp), (-1, xr), (-1, xdiff)], 1));
            lin_w.push(lin_witness(&[(1, yr), (1, yp), (-1, rhsy)], -1));
        }

        InstanceData {
            elems,
            mul_eqs: muls.try_into().ok().unwrap(),
            lin_witnesses: lin_w.try_into().ok().unwrap(),
        }
    }

    fn expr_from_bits<AB: AirBuilder>(
        row: &[AB::Var],
        start: usize,
        bit_count: usize,
    ) -> AB::Expr {
        let mut acc = AB::Expr::ZERO;
        let mut weight = AB::Expr::ONE;
        for b in 0..bit_count {
            let bit: AB::Expr = row[start + b].into();
            acc = acc + bit * weight.clone();
            weight = weight.clone() + weight;
        }
        acc
    }

    fn elem_limb_expr<AB: AirBuilder>(
        row: &[AB::Var],
        elem: usize,
        limb: usize,
    ) -> AB::Expr {
        row[elem_limb_col(elem, limb)].into()
    }

    fn lin_k_expr<AB: AirBuilder>(row: &[AB::Var], eq: usize) -> AB::Expr {
        expr_from_bits::<AB>(row, lin_k_bit_col(eq, 0), LIN_K_BITS)
    }

    fn lin_carry_expr<AB: AirBuilder>(
        row: &[AB::Var],
        eq: usize,
        limb: usize,
    ) -> AB::Expr {
        expr_from_bits::<AB>(
            row,
            lin_carry_bit_col(eq, limb, 0),
            LIN_CARRY_BITS,
        ) - AB::Expr::from_u64(LIN_CARRY_OFFSET as u64)
    }

    fn mul_limb_from_bits<AB: AirBuilder>(
        row: &[AB::Var],
        bits_start: usize,
        limb: usize,
    ) -> AB::Expr {
        expr_from_bits::<AB>(row, bits_start + limb * LIMB_BITS, LIMB_BITS)
    }

    fn mul_signed_carry_expr<AB: AirBuilder>(
        row: &[AB::Var],
        idx: usize,
    ) -> AB::Expr {
        let sign: AB::Expr = row[MUL_CARRY_SIGN_START + idx].into();
        let mag = expr_from_bits::<AB>(
            row,
            MUL_CARRY_BITS_START + idx * MUL_CARRY_BITS,
            MUL_CARRY_BITS,
        );
        (AB::Expr::ONE - sign.clone() - sign) * mag
    }

    fn add_term<AB: AirBuilder>(
        acc: AB::Expr,
        term: AB::Expr,
        coeff: i64,
    ) -> AB::Expr {
        match coeff {
            1 => acc + term,
            -1 => acc - term,
            _ => panic!("unsupported coefficient"),
        }
    }

    fn linear_spec(i: usize, j: usize) -> (Vec<(i64, usize)>, i64) {
        let xp = d_x_elem(i);
        let yp = d_y_elem(i);
        let xq = pub_x_elem(2 * i + 1);
        let yq = pub_y_elem(2 * i + 1);
        let xr = pub_x_elem(2 * i);
        let yr = pub_y_elem(2 * i);

        match j {
            0 => (vec![(1, xq), (-1, xp), (-1, inter_elem(i, I_DX))], 1),
            1 => (vec![(1, yq), (-1, yp), (-1, inter_elem(i, I_DY))], 1),
            2 => (vec![(1, xr), (1, xp), (1, xq), (-1, inter_elem(i, I_RHSX))], -1),
            3 => (vec![(1, xp), (-1, xr), (-1, inter_elem(i, I_XDIFF))], 1),
            4 => (vec![(1, yr), (1, yp), (-1, inter_elem(i, I_RHSY))], -1),
            _ => unreachable!(),
        }
    }

    fn mul_spec(eq: usize) -> (usize, usize, usize) {
        let i = eq / 3;
        match eq % 3 {
            0 => (
                inter_elem(i, I_LAMBDA),
                inter_elem(i, I_DX),
                inter_elem(i, I_DY),
            ),
            1 => (
                inter_elem(i, I_LAMBDA),
                inter_elem(i, I_LAMBDA),
                inter_elem(i, I_RHSX),
            ),
            2 => (
                inter_elem(i, I_LAMBDA),
                inter_elem(i, I_XDIFF),
                inter_elem(i, I_RHSY),
            ),
            _ => unreachable!(),
        }
    }



    fn fill_elem(row: &mut [Val], elem: usize, limbs: &[u64; LIMBS]) {
        for l in 0..LIMBS {
            row[elem_limb_col(elem, l)] = Val::from_u64(limbs[l]);
            for b in 0..LIMB_BITS {
                row[elem_bits_col(elem, l, b)] =
                    Val::from_u64((limbs[l] >> b) & 1);
            }
        }
    }

    fn encode_mul_row(
        row: &mut [Val],
        eq_index: usize,
        eq: &MulEquation,
    ) {
        for i in 0..NUM_MUL_EQS {
            row[SEL_START + i] = if i == eq_index { Val::ONE } else { Val::ZERO };
        }

        let a = fq_to_limbs(&eq.a);
        let b = fq_to_limbs(&eq.b);
        let c = fq_to_limbs(&eq.c);
        let w = build_mul_witness(eq);

        for l in 0..LIMBS {
            row[A_START + l] = Val::from_u64(a[l]);
            row[B_START + l] = Val::from_u64(b[l]);
            row[C_START + l] = Val::from_u64(c[l]);

            for bit in 0..LIMB_BITS {
                row[A_BITS_START + l * LIMB_BITS + bit] = Val::from_u64((a[l] >> bit) & 1);
                row[B_BITS_START + l * LIMB_BITS + bit] = Val::from_u64((b[l] >> bit) & 1);
                row[C_BITS_START + l * LIMB_BITS + bit] = Val::from_u64((c[l] >> bit) & 1);
                row[K_BITS_START + l * LIMB_BITS + bit] =
                    Val::from_u64((w.k_limbs[l] >> bit) & 1);
            }
        }

        for i in 0..CARRY_COUNT {
            row[MUL_CARRY_SIGN_START + i] =
                if w.carry_signs[i] { Val::ONE } else { Val::ZERO };
            for b in 0..MUL_CARRY_BITS {
                row[MUL_CARRY_BITS_START + i * MUL_CARRY_BITS + b] =
                    Val::from_u64((w.carry_mags[i] >> b) & 1);
            }
        }
    }

    fn make_trace(data: &InstanceData) -> RowMajorMatrix<Val> {
        let height = NUM_MUL_EQS.next_power_of_two();
        let mut values = vec![Val::ZERO; height * WIDTH];

        for r in 0..height {
            let row = &mut values[r * WIDTH..(r + 1) * WIDTH];

            for e in 0..GLOBAL_ELEMS {
                fill_elem(row, e, &data.elems[e]);
            }

            if r == 0 {
                for eq in 0..NUM_LINEAR_EQS {
                    let w = &data.lin_witnesses[eq];
                    for b in 0..LIN_K_BITS {
                        row[lin_k_bit_col(eq, b)] = Val::from_u64((w.k >> b) & 1);
                    }
                    for l in 0..LIMBS {
                        for b in 0..LIN_CARRY_BITS {
                            row[lin_carry_bit_col(eq, l, b)] =
                                Val::from_u64((w.carries_shifted[l] >> b) & 1);
                        }
                    }
                }
            }

            let eq_index = if r < NUM_MUL_EQS { r } else { NUM_MUL_EQS - 1 };
            encode_mul_row(row, eq_index, &data.mul_eqs[eq_index]);
        }

        RowMajorMatrix::new(values, WIDTH)
    }

    pub(super) fn eval<AB: AirBuilder>(
        builder: &mut AB,
        local: &[AB::Var],
        next: &[AB::Var],
        public_values: &[AB::Expr],
    ) {
    // Bind the six public points on the first row.
            for p in 0..NUM_PUBLIC_POINTS {
                for l in 0..LIMBS {
                    builder.when_first_row().assert_eq(
                        local[elem_limb_col(pub_x_elem(p), l)],
                        public_values[p * POINT_LIMBS + l].clone(),
                    );
                    builder.when_first_row().assert_eq(
                        local[elem_limb_col(pub_y_elem(p), l)],
                        public_values[p * POINT_LIMBS + LIMBS + l].clone(),
                    );
                }
            }

            // Carry every global field element through the trace.
            for e in 0..GLOBAL_ELEMS {
                for l in 0..LIMBS {
                    builder.when_transition().assert_eq(
                        local[elem_limb_col(e, l)],
                        next[elem_limb_col(e, l)],
                    );
                }
            }

            // First-row range checks for all global field elements.
            for e in 0..GLOBAL_ELEMS {
                for l in 0..LIMBS {
                    let mut rec = AB::Expr::ZERO;
                    let mut w = AB::Expr::ONE;
                    for b in 0..LIMB_BITS {
                        let bit: AB::Expr = local[elem_bits_col(e, l, b)].into();
                        builder.when_first_row().assert_zero(
                            bit.clone() * (bit.clone() - AB::Expr::ONE)
                        );
                        rec = rec + bit * w.clone();
                        w = w.clone() + w;
                    }
                    builder.when_first_row().assert_eq(
                        local[elem_limb_col(e, l)],
                        rec,
                    );
                }
            }

            // Dynamic modular linear relations used to derive dx,dy,rhsx,xdiff,rhsy.
            for i in 0..NUM_RELATIONS {
                for j in 0..5 {
                    let eq = i * 5 + j;
                    let (terms, q_coeff) = linear_spec(i, j);

                    for b in 0..LIN_K_BITS {
                        let bit: AB::Expr = local[lin_k_bit_col(eq, b)].into();
                        builder.when_first_row().assert_zero(
                            bit.clone() * (bit - AB::Expr::ONE)
                        );
                    }
                    for l in 0..LIMBS {
                        for b in 0..LIN_CARRY_BITS {
                            let bit: AB::Expr = local[lin_carry_bit_col(eq, l, b)].into();
                            builder.when_first_row().assert_zero(
                                bit.clone() * (bit - AB::Expr::ONE)
                            );
                        }
                    }

                    let k = lin_k_expr::<AB>(local, eq);

                    for t in 0..LIMBS {
                        let mut lhs = if t == 0 {
                            AB::Expr::ZERO
                        } else {
                            lin_carry_expr::<AB>(local, eq, t - 1)
                        };

                        for (coeff, elem) in &terms {
                            lhs = add_term::<AB>(
                                lhs,
                                elem_limb_expr::<AB>(local, *elem, t),
                                *coeff,
                            );
                        }

                        let qterm = k.clone() * AB::Expr::from_u64(Q_LIMBS[t]);
                        lhs = if q_coeff == 1 { lhs + qterm } else { lhs - qterm };

                        let next_carry = lin_carry_expr::<AB>(local, eq, t);
                        builder.when_first_row().assert_zero(
                            lhs - next_carry * AB::Expr::from_u64(BASE)
                        );
                    }

                    builder.when_first_row().assert_zero(
                        lin_carry_expr::<AB>(local, eq, LIMBS - 1)
                    );
                }
            }

            // One-hot selector for the nine modular multiplication equations.
            let mut sel_sum = AB::Expr::ZERO;
            for i in 0..NUM_MUL_EQS {
                let s: AB::Expr = local[SEL_START + i].into();
                builder.assert_zero(s.clone() * (s.clone() - AB::Expr::ONE));
                sel_sum = sel_sum + s;
            }
            builder.assert_one(sel_sum);

            // Range checks for multiplication gadget.
            for l in 0..LIMBS {
                let a_rec = mul_limb_from_bits::<AB>(local, A_BITS_START, l);
                let b_rec = mul_limb_from_bits::<AB>(local, B_BITS_START, l);
                let c_rec = mul_limb_from_bits::<AB>(local, C_BITS_START, l);

                builder.assert_eq(local[A_START + l], a_rec);
                builder.assert_eq(local[B_START + l], b_rec);
                builder.assert_eq(local[C_START + l], c_rec);

                for b in 0..LIMB_BITS {
                    for start in [A_BITS_START, B_BITS_START, C_BITS_START, K_BITS_START] {
                        let bit: AB::Expr = local[start + l * LIMB_BITS + b].into();
                        builder.assert_zero(bit.clone() * (bit - AB::Expr::ONE));
                    }
                }
            }

            for i in 0..CARRY_COUNT {
                let sign: AB::Expr = local[MUL_CARRY_SIGN_START + i].into();
                builder.assert_zero(sign.clone() * (sign - AB::Expr::ONE));
                for b in 0..MUL_CARRY_BITS {
                    let bit: AB::Expr =
                        local[MUL_CARRY_BITS_START + i * MUL_CARRY_BITS + b].into();
                    builder.assert_zero(bit.clone() * (bit - AB::Expr::ONE));
                }
            }

            // Bind multiplication a,b,c limbs to dynamic global intermediates.
            for l in 0..LIMBS {
                let mut ea = AB::Expr::ZERO;
                let mut eb = AB::Expr::ZERO;
                let mut ec = AB::Expr::ZERO;

                for eq in 0..NUM_MUL_EQS {
                    let s: AB::Expr = local[SEL_START + eq].into();
                    let (ae, be, ce) = mul_spec(eq);
                    ea = ea + s.clone() * elem_limb_expr::<AB>(local, ae, l);
                    eb = eb + s.clone() * elem_limb_expr::<AB>(local, be, l);
                    ec = ec + s * elem_limb_expr::<AB>(local, ce, l);
                }

                builder.assert_eq(local[A_START + l], ea);
                builder.assert_eq(local[B_START + l], eb);
                builder.assert_eq(local[C_START + l], ec);
            }

            // Enforce a*b-c=k*q.
            for t in 0..32 {
                let mut lhs = if t == 0 {
                    AB::Expr::ZERO
                } else {
                    mul_signed_carry_expr::<AB>(local, t - 1)
                };

                for i in 0..LIMBS {
                    if t >= i {
                        let j = t - i;
                        if j < LIMBS {
                            let ai: AB::Expr = local[A_START + i].into();
                            let bj: AB::Expr = local[B_START + j].into();
                            lhs = lhs + ai * bj;
                        }
                    }
                }

                for i in 0..LIMBS {
                    if t >= i {
                        let j = t - i;
                        if j < LIMBS {
                            let ki = mul_limb_from_bits::<AB>(local, K_BITS_START, i);
                            lhs = lhs - ki * AB::Expr::from_u64(Q_LIMBS[j]);
                        }
                    }
                }

                if t < LIMBS {
                    let ci: AB::Expr = local[C_START + t].into();
                    lhs = lhs - ci;
                }

                let next_carry = if t < CARRY_COUNT {
                    mul_signed_carry_expr::<AB>(local, t)
                } else {
                    AB::Expr::ZERO
                };

                builder.assert_zero(
                    lhs - next_carry * AB::Expr::from_u64(BASE)
                );
            }
    
    }


    pub(super) fn make_values_with_height(
        public_points: &[G1Affine; NUM_PUBLIC_POINTS],
        d_points: &[G1Affine; NUM_RELATIONS],
        height: usize,
    ) -> Vec<Val> {
        assert!(height.is_power_of_two());
        assert!(height >= NUM_MUL_EQS.next_power_of_two());

        let data = make_instance(public_points, d_points);
        let mut values = vec![Val::ZERO; height * WIDTH];

        for r in 0..height {
            let row = &mut values[r * WIDTH..(r + 1) * WIDTH];

            for e in 0..GLOBAL_ELEMS {
                fill_elem(row, e, &data.elems[e]);
            }

            if r == 0 {
                for eq in 0..NUM_LINEAR_EQS {
                    let w = &data.lin_witnesses[eq];

                    for b in 0..LIN_K_BITS {
                        row[lin_k_bit_col(eq, b)] =
                            Val::from_u64((w.k >> b) & 1);
                    }

                    for l in 0..LIMBS {
                        for b in 0..LIN_CARRY_BITS {
                            row[lin_carry_bit_col(eq, l, b)] =
                                Val::from_u64(
                                    (w.carries_shifted[l] >> b) & 1
                                );
                        }
                    }
                }
            }

            let eq_index =
                if r < NUM_MUL_EQS {
                    r
                } else {
                    NUM_MUL_EQS - 1
                };

            encode_mul_row(
                row,
                eq_index,
                &data.mul_eqs[eq_index],
            );
        }

        values
    }

}


// ============================================================
// Dynamic Miller-line evaluation layer
//
// On every Miller-line row this layer proves, over BN254 Fq:
//
//     line.c0.c0 = raw.c0.c0 * y_P
//     line.c0.c1 = raw.c0.c1 * y_P
//     line.c3.c0 = raw.c3.c0 * x_P
//     line.c3.c1 = raw.c3.c1 * x_P
//
// and binds line.c4 directly to the raw prepared G2 coefficient.
//
// P is selected by the Miller step:
//     0 -> internal D_1
//     1 -> internal D_2
//     2 -> internal D_3
//     3 -> public W
//
// This removes the last host-side G1 Miller-line evaluation dependency.
// ============================================================
mod line_eval_layer {
    use super::*;

    const MULS: usize = 4;
    const LCARRY_COUNT: usize = 31;
    const LCARRY_BITS: usize = 48;

    // Per multiplication slot:
    //   A,B,C limbs
    //   A,B,C,K bit decompositions
    //   signed carry witness
    const A_START: usize = 0;
    const B_START: usize = A_START + LIMBS;
    const C_START: usize = B_START + LIMBS;
    const A_BITS_START: usize = C_START + LIMBS;
    const B_BITS_START: usize = A_BITS_START + LIMBS * LIMB_BITS;
    const C_BITS_START: usize = B_BITS_START + LIMBS * LIMB_BITS;
    const K_BITS_START: usize = C_BITS_START + LIMBS * LIMB_BITS;
    const CARRY_SIGN_START: usize = K_BITS_START + LIMBS * LIMB_BITS;
    const CARRY_BITS_START: usize = CARRY_SIGN_START + LCARRY_COUNT;
    const SLOT_WIDTH: usize = CARRY_BITS_START + LCARRY_COUNT * LCARRY_BITS;

    pub(super) const WIDTH: usize = MULS * SLOT_WIDTH;

    #[derive(Clone)]
    struct MulWitness {
        k_limbs: [u64; LIMBS],
        carry_signs: [bool; LCARRY_COUNT],
        carry_mags: [u64; LCARRY_COUNT],
    }

    fn slot_start(slot: usize) -> usize { slot * SLOT_WIDTH }

    fn a_col(slot: usize, limb: usize) -> usize {
        slot_start(slot) + A_START + limb
    }
    fn b_col(slot: usize, limb: usize) -> usize {
        slot_start(slot) + B_START + limb
    }
    fn c_col(slot: usize, limb: usize) -> usize {
        slot_start(slot) + C_START + limb
    }
    fn bits_col(slot: usize, start: usize, limb: usize, bit: usize) -> usize {
        slot_start(slot) + start + limb * LIMB_BITS + bit
    }
    fn carry_sign_col(slot: usize, c: usize) -> usize {
        slot_start(slot) + CARRY_SIGN_START + c
    }
    fn carry_bit_col(slot: usize, c: usize, bit: usize) -> usize {
        slot_start(slot) + CARRY_BITS_START + c * LCARRY_BITS + bit
    }

    fn fq_from_limbs(limbs: &[u64; LIMBS]) -> Fq {
        let mut bytes = Vec::with_capacity(LIMBS * 2);
        for limb in limbs {
            bytes.push((*limb & 0xff) as u8);
            bytes.push((*limb >> 8) as u8);
        }
        Fq::from_le_bytes_mod_order(&bytes)
    }

    fn build_mul_witness(a: Fq, b: Fq, c: Fq) -> MulWitness {
        let a_big = fq_to_biguint(&a);
        let b_big = fq_to_biguint(&b);
        let c_big = fq_to_biguint(&c);
        let q_big = modulus_biguint();

        let prod = &a_big * &b_big;
        assert!(prod >= c_big);
        let diff = &prod - &c_big;
        assert_eq!(&diff % &q_big, BigUint::from(0u64));

        let k_big = &diff / &q_big;
        let k_limbs = biguint_to_limbs::<LIMBS>(&k_big);

        let a_limbs = fq_to_limbs(&a);
        let b_limbs = fq_to_limbs(&b);
        let c_limbs = fq_to_limbs(&c);

        let mut carry_signs = [false; LCARRY_COUNT];
        let mut carry_mags = [0u64; LCARRY_COUNT];
        let mut carry: i128 = 0;

        for t in 0..32 {
            let mut x = carry;

            for i in 0..LIMBS {
                if t >= i {
                    let j = t - i;
                    if j < LIMBS {
                        x += a_limbs[i] as i128 * b_limbs[j] as i128;
                    }
                }
            }

            for i in 0..LIMBS {
                if t >= i {
                    let j = t - i;
                    if j < LIMBS {
                        x -= k_limbs[i] as i128 * Q_LIMBS[j] as i128;
                    }
                }
            }

            if t < LIMBS {
                x -= c_limbs[t] as i128;
            }

            assert_eq!(x % BASE as i128, 0);
            let next = x / BASE as i128;

            if t < LCARRY_COUNT {
                let neg = next < 0;
                let mag = next.unsigned_abs();
                assert!(mag < (1u128 << LCARRY_BITS));
                carry_signs[t] = neg;
                carry_mags[t] = mag as u64;
            } else {
                assert_eq!(next, 0);
            }

            carry = next;
        }

        MulWitness { k_limbs, carry_signs, carry_mags }
    }

    fn fill_bits(row: &mut [Val], col: usize, value: u64, bits: usize) {
        for b in 0..bits {
            row[col + b] = Val::from_u64((value >> b) & 1);
        }
    }

    fn encode_slot(row: &mut [Val], slot: usize, a: Fq, b: Fq, c: Fq) {
        let al = fq_to_limbs(&a);
        let bl = fq_to_limbs(&b);
        let cl = fq_to_limbs(&c);
        let w = build_mul_witness(a, b, c);

        for l in 0..LIMBS {
            row[a_col(slot,l)] = Val::from_u64(al[l]);
            row[b_col(slot,l)] = Val::from_u64(bl[l]);
            row[c_col(slot,l)] = Val::from_u64(cl[l]);

            fill_bits(row, bits_col(slot,A_BITS_START,l,0), al[l], LIMB_BITS);
            fill_bits(row, bits_col(slot,B_BITS_START,l,0), bl[l], LIMB_BITS);
            fill_bits(row, bits_col(slot,C_BITS_START,l,0), cl[l], LIMB_BITS);
            fill_bits(row, bits_col(slot,K_BITS_START,l,0), w.k_limbs[l], LIMB_BITS);
        }

        for cidx in 0..LCARRY_COUNT {
            row[carry_sign_col(slot,cidx)] =
                if w.carry_signs[cidx] { Val::ONE } else { Val::ZERO };
            fill_bits(
                row,
                carry_bit_col(slot,cidx,0),
                w.carry_mags[cidx],
                LCARRY_BITS,
            );
        }
    }

    pub(super) fn make_values_with_height(
        schedule: &[NativeRow],
        constants: &[MillerStepConst],
        pairs: &[(G1Affine, G2Affine); 4],
        height: usize,
    ) -> Vec<Val> {
        let mut values = vec![Val::ZERO; height * WIDTH];

        for r in 0..height {
            if r >= schedule.len() { continue; }
            let native = &schedule[r];
            if native.op != Op::MillerLine { continue; }

            let step = native.miller_step.unwrap();
            let mc = &constants[step];
            let p = pairs[mc.pair_index].0;

            let raw = &mc.raw_coeff_limbs;
            let a0 = fq_from_limbs(raw[0*LIMBS..1*LIMBS].try_into().unwrap());
            let a1 = fq_from_limbs(raw[1*LIMBS..2*LIMBS].try_into().unwrap());
            let a2 = fq_from_limbs(raw[2*LIMBS..3*LIMBS].try_into().unwrap());
            let a3 = fq_from_limbs(raw[3*LIMBS..4*LIMBS].try_into().unwrap());

            let line = native.line.as_ref().unwrap();
            let cs = [line.0.c0, line.0.c1, line.1.c0, line.1.c1];
            let as_ = [a0,a1,a2,a3];
            let bs = [p.y,p.y,p.x,p.x];

            let row = &mut values[r*WIDTH..(r+1)*WIDTH];
            for slot in 0..MULS {
                encode_slot(row, slot, as_[slot], bs[slot], cs[slot]);
            }
        }

        values
    }

    fn expr_from_bits<AB: AirBuilder>(
        row: &[AB::Var],
        start: usize,
        bit_count: usize,
    ) -> AB::Expr {
        let mut acc = AB::Expr::ZERO;
        let mut weight = AB::Expr::ONE;
        for b in 0..bit_count {
            let bit: AB::Expr = row[start+b].into();
            acc = acc + bit * weight.clone();
            weight = weight.clone() + weight;
        }
        acc
    }

    fn signed_carry<AB: AirBuilder>(row: &[AB::Var], slot: usize, c: usize) -> AB::Expr {
        let sign: AB::Expr = row[carry_sign_col(slot,c)].into();
        let mag = expr_from_bits::<AB>(row, carry_bit_col(slot,c,0), LCARRY_BITS);
        (AB::Expr::ONE - sign.clone() - sign) * mag
    }

    fn point_coord_limb<AB: AirBuilder>(
        diff_local: &[AB::Var],
        public_values: &[AB::Expr],
        pair_index: usize,
        use_y: bool,
        limb: usize,
    ) -> AB::Expr {
        if pair_index < 3 {
            let elem = if use_y {
                diff_layer::d_y_elem(pair_index)
            } else {
                diff_layer::d_x_elem(pair_index)
            };
            diff_local[diff_layer::elem_limb_col(elem, limb)].into()
        } else {
            // Public point 6 is W.
            let base = 6 * G1_POINT_LIMBS;
            let off = if use_y { LIMBS } else { 0 };
            public_values[base + off + limb].clone()
        }
    }

    pub(super) fn eval<AB: AirBuilder>(
        builder: &mut AB,
        row: &[AB::Var],
        pair_local: &[AB::Var],
        diff_local: &[AB::Var],
        public_values: &[AB::Expr],
        miller: &[MillerStepConst],
        mline: AB::Expr,
    ) {
        // Range-check all four modular-multiplication witnesses only when the
        // row is a Miller-line row.
        for slot in 0..MULS {
            for l in 0..LIMBS {
                for (vcol,bstart) in [
                    (a_col(slot,l), bits_col(slot,A_BITS_START,l,0)),
                    (b_col(slot,l), bits_col(slot,B_BITS_START,l,0)),
                    (c_col(slot,l), bits_col(slot,C_BITS_START,l,0)),
                ] {
                    let rec = expr_from_bits::<AB>(row,bstart,LIMB_BITS);
                    builder.when(mline.clone()).assert_eq(row[vcol],rec);
                    for b in 0..LIMB_BITS {
                        let bit: AB::Expr = row[bstart+b].into();
                        builder.when(mline.clone()).assert_zero(
                            bit.clone()*(bit-AB::Expr::ONE)
                        );
                    }
                }

                let kstart = bits_col(slot,K_BITS_START,l,0);
                for b in 0..LIMB_BITS {
                    let bit: AB::Expr = row[kstart+b].into();
                    builder.when(mline.clone()).assert_zero(
                        bit.clone()*(bit-AB::Expr::ONE)
                    );
                }
            }

            for c in 0..LCARRY_COUNT {
                let sign: AB::Expr = row[carry_sign_col(slot,c)].into();
                builder.when(mline.clone()).assert_zero(
                    sign.clone()*(sign-AB::Expr::ONE)
                );
                for b in 0..LCARRY_BITS {
                    let bit: AB::Expr = row[carry_bit_col(slot,c,b)].into();
                    builder.when(mline.clone()).assert_zero(
                        bit.clone()*(bit-AB::Expr::ONE)
                    );
                }
            }
        }

        // Bind A (raw G2 prepared coefficient), B (the selected G1 x/y),
        // and C (the evaluated sparse line already consumed by the Fq12 AIR).
        for slot in 0..MULS {
            let raw_component = slot; // c0.c0,c0.c1,c3.c0,c3.c1
            let use_y = slot < 2;

            for l in 0..LIMBS {
                let mut expected_a = AB::Expr::ZERO;
                let mut expected_b = AB::Expr::ZERO;

                for j in 0..MILLER_STEPS {
                    if miller[j].is_square { continue; }
                    let id: AB::Expr = pair_local[MILLER_ID_START+j].into();
                    let aval = miller[j].raw_coeff_limbs[raw_component*LIMBS+l];
                    if aval != 0 {
                        expected_a = expected_a + id.clone()*AB::Expr::from_u64(aval);
                    }
                    expected_b = expected_b + id * point_coord_limb::<AB>(
                        diff_local,
                        public_values,
                        miller[j].pair_index,
                        use_y,
                        l,
                    );
                }

                builder.when(mline.clone()).assert_eq(row[a_col(slot,l)], expected_a);
                builder.when(mline.clone()).assert_eq(row[b_col(slot,l)], expected_b);
                builder.when(mline.clone()).assert_eq(
                    row[c_col(slot,l)],
                    pair_local[LINE_START + raw_component*LIMBS + l],
                );
            }
        }

        // c4 is independent of P, so bind both Fq components directly to the
        // raw prepared coefficient selected by the Miller step.
        for comp in 4..6 {
            for l in 0..LIMBS {
                let mut expected = AB::Expr::ZERO;
                for j in 0..MILLER_STEPS {
                    if miller[j].is_square { continue; }
                    let val = miller[j].raw_coeff_limbs[comp*LIMBS+l];
                    if val != 0 {
                        let id: AB::Expr = pair_local[MILLER_ID_START+j].into();
                        expected = expected + id*AB::Expr::from_u64(val);
                    }
                }
                builder.when(mline.clone()).assert_eq(
                    pair_local[LINE_START + comp*LIMBS + l],
                    expected,
                );
            }
        }

        // Prove a*b-c = k*q for the four Fq products.
        for slot in 0..MULS {
            for t in 0..32 {
                let mut lhs = if t == 0 {
                    AB::Expr::ZERO
                } else {
                    signed_carry::<AB>(row,slot,t-1)
                };

                for i in 0..LIMBS {
                    if t >= i {
                        let j=t-i;
                        if j<LIMBS {
                            let ai: AB::Expr = row[a_col(slot,i)].into();
                            let bj: AB::Expr = row[b_col(slot,j)].into();
                            lhs = lhs + ai*bj;
                        }
                    }
                }

                for i in 0..LIMBS {
                    if t >= i {
                        let j=t-i;
                        if j<LIMBS {
                            let ki = expr_from_bits::<AB>(
                                row,
                                bits_col(slot,K_BITS_START,i,0),
                                LIMB_BITS,
                            );
                            lhs = lhs - ki*AB::Expr::from_u64(Q_LIMBS[j]);
                        }
                    }
                }

                if t<LIMBS {
                    let ci: AB::Expr = row[c_col(slot,t)].into();
                    lhs = lhs-ci;
                }

                let next_carry = if t<LCARRY_COUNT {
                    signed_carry::<AB>(row,slot,t)
                } else {
                    AB::Expr::ZERO
                };

                builder.when(mline.clone()).assert_zero(
                    lhs-next_carry*AB::Expr::from_u64(BASE)
                );
            }
        }
    }
}

// ============================================================
// AIR
// ============================================================

struct PairingAir {
    miller: Vec<MillerStepConst>,
    pairing_output_limbs: [u64; FQ12_LIMBS],
    final_exp_limbs: [u64; EXP_LIMBS],

    // Canonical verifier-visible monolithic statement:
    //
    //   C_f1,C_F1,C_f2,C_F2,C_f3,C_F3,W.
    //
    // Pairing constants are deterministically generated by the verifier from
    // this same public statement.  The difference sub-AIR simultaneously
    // proves D_i = C_fi - C_Fi inside the same STARK proof.
    statement_g1_limbs: [u64; NUM_PUBLIC_VALUES],
}

impl<F> BaseAir<F> for PairingAir {
    fn width(&self) -> usize {
        PAIR_WIDTH + line_eval_layer::WIDTH + diff_layer::WIDTH
    }

    fn num_public_values(&self) -> usize {
        NUM_PUBLIC_VALUES
    }
}

impl<AB: AirBuilder> Air<AB> for PairingAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let next = main.next_slice();

        let public_values: Vec<AB::Expr> = {
            let pis = builder.public_values();
            pis.iter().map(|v| (*v).into()).collect()
        };

        // ----------------------------------------------------
        // Public monolithic statement binding.
        //
        // Public point order:
        //   0 = C_f1, 1 = C_F1,
        //   2 = C_f2, 3 = C_F2,
        //   4 = C_f3, 5 = C_F3,
        //   6 = W.
        //
        // D_i do not appear in the public input.
        // ----------------------------------------------------
        for i in 0..NUM_PUBLIC_VALUES {
            builder.when_first_row().assert_eq(
                public_values[i].clone(),
                AB::Expr::from_u64(self.statement_g1_limbs[i]),
            );
        }

        // ----------------------------------------------------
        // Selectors: boolean + exactly one.
        // ----------------------------------------------------
        let mut selector_sum = AB::Expr::ZERO;

        for i in 0..NUM_SELECTORS {
            let s: AB::Expr = local[i].into();
            builder.assert_zero(
                s.clone() * (s.clone() - AB::Expr::ONE)
            );
            selector_sum = selector_sum + s;
        }

        builder.assert_one(selector_sum);

        let msq: AB::Expr = local[SEL_MILLER_SQ].into();
        let mline: AB::Expr = local[SEL_MILLER_LINE].into();
        let einit: AB::Expr = local[SEL_EXP_INIT].into();
        let eacc: AB::Expr = local[SEL_EXP_ACC].into();
        let ebase: AB::Expr = local[SEL_EXP_BASE].into();
        let output: AB::Expr = local[SEL_OUTPUT].into();
        let pad: AB::Expr = local[SEL_PAD].into();

        let miller_active = msq.clone() + mline.clone();

        // First row is Miller step 0.
        builder.when_first_row().assert_one(
            local[MILLER_ID_START]
        );

        // ----------------------------------------------------
        // Range-check ACC, BASE, EXP limbs.
        // ----------------------------------------------------
        for li in 0..FQ12_LIMBS {
            let limb: AB::Expr = local[ACC_START + li].into();
            let bs = ACC_BITS_START + li * LIMB_BITS;
            let rec = expr_from_bits::<AB>(local, bs, LIMB_BITS);
            builder.assert_eq(limb, rec);

            for b in 0..LIMB_BITS {
                let bit: AB::Expr = local[bs + b].into();
                builder.assert_zero(
                    bit.clone() * (bit - AB::Expr::ONE)
                );
            }
        }

        for li in 0..FQ12_LIMBS {
            let limb: AB::Expr = local[BASE_START + li].into();
            let bs = BASE_BITS_START + li * LIMB_BITS;
            let rec = expr_from_bits::<AB>(local, bs, LIMB_BITS);
            builder.assert_eq(limb, rec);

            for b in 0..LIMB_BITS {
                let bit: AB::Expr = local[bs + b].into();
                builder.assert_zero(
                    bit.clone() * (bit - AB::Expr::ONE)
                );
            }
        }

        for li in 0..EXP_LIMBS {
            let limb: AB::Expr = local[EXP_START + li].into();
            let bs = EXP_BITS_START + li * LIMB_BITS;
            let rec = expr_from_bits::<AB>(local, bs, LIMB_BITS);
            builder.assert_eq(limb, rec);

            for b in 0..LIMB_BITS {
                let bit: AB::Expr = local[bs + b].into();
                builder.assert_zero(
                    bit.clone() * (bit - AB::Expr::ONE)
                );
            }
        }

        let exp_bit: AB::Expr = local[EXP_BITS_START].into();

        // ----------------------------------------------------
        // Exact Miller schedule one-hot.
        // ----------------------------------------------------
        let mut mid_sum = AB::Expr::ZERO;
        let mut mid_weighted = AB::Expr::ZERO;
        let mut expected_sq = AB::Expr::ZERO;
        let mut expected_coeff = AB::Expr::ZERO;

        for j in 0..MILLER_STEPS {
            let id: AB::Expr =
                local[MILLER_ID_START + j].into();

            builder.assert_zero(
                id.clone() * (id.clone() - AB::Expr::ONE)
            );

            mid_sum = mid_sum + id.clone();

            mid_weighted =
                mid_weighted
                + id.clone() * AB::Expr::from_u64(j as u64);

            if self.miller[j].is_square {
                expected_sq = expected_sq + id.clone();
            }

            expected_coeff =
                expected_coeff
                + id * AB::Expr::from_u64(
                    self.miller[j].coeff_index as u64
                );
        }

        builder.assert_eq(mid_sum, miller_active.clone());

        builder.assert_eq(
            local[MILLER_STEP_INDEX],
            mid_weighted,
        );

        builder.assert_eq(msq.clone(), expected_sq);

        builder
            .when(miller_active.clone())
            .assert_eq(
                local[COEFF_INDEX],
                expected_coeff,
            );

        // Evaluated Miller-line limbs are derived dynamically below.

        // Miller state begins at f=1, BASE=0, EXP=0.
        builder.when_first_row().assert_one(local[ACC_START]);

        for i in (ACC_START + 1)..ACC_END {
            builder.when_first_row().assert_zero(local[i]);
        }

        // Exact Miller step progression.
        let next_msq: AB::Expr = next[SEL_MILLER_SQ].into();
        let next_mline: AB::Expr = next[SEL_MILLER_LINE].into();
        let next_einit: AB::Expr = next[SEL_EXP_INIT].into();
        let next_miller = next_msq + next_mline;

        builder
            .when_transition()
            .when(miller_active.clone() * next_miller.clone())
            .assert_eq(
                next[MILLER_STEP_INDEX],
                local[MILLER_STEP_INDEX] + AB::Expr::ONE,
            );

        builder
            .when_transition()
            .when(miller_active.clone() * next_einit.clone())
            .assert_eq(
                local[MILLER_STEP_INDEX],
                AB::Expr::from_u64((MILLER_STEPS - 1) as u64),
            );

        builder
            .when_transition()
            .when(miller_active.clone())
            .assert_one(next_miller + next_einit);

        // Miller rows keep BASE/EXP zero.
        for i in BASE_START..BASE_END {
            builder.when(miller_active.clone()).assert_zero(local[i]);
        }
        for i in EXP_START..EXP_END {
            builder.when(miller_active.clone()).assert_zero(local[i]);
        }

        // ----------------------------------------------------
        // ExpInit:
        //
        // next.acc  = 1
        // next.base = Miller output
        // next.exp  = fixed final exponent
        // next.counter = 0
        // ----------------------------------------------------
        let next_eacc: AB::Expr = next[SEL_EXP_ACC].into();

        builder
            .when_transition()
            .when(einit.clone())
            .assert_one(next_eacc.clone());

        builder
            .when_transition()
            .when(einit.clone())
            .assert_one(next[ACC_START]);

        for i in (ACC_START + 1)..ACC_END {
            builder
                .when_transition()
                .when(einit.clone())
                .assert_zero(next[i]);
        }

        for i in 0..FQ12_LIMBS {
            builder
                .when_transition()
                .when(einit.clone())
                .assert_eq(
                    next[BASE_START + i],
                    local[ACC_START + i],
                );
        }

        for i in 0..EXP_LIMBS {
            builder
                .when_transition()
                .when(einit.clone())
                .assert_eq(
                    next[EXP_START + i],
                    AB::Expr::from_u64(self.final_exp_limbs[i]),
                );
        }

        builder
            .when_transition()
            .when(einit.clone())
            .assert_zero(next[EXP_COUNTER]);

        // ----------------------------------------------------
        // ExpAcc:
        //
        // bit=0: next.acc=acc
        // bit=1: generic multiplication proves acc*base
        // base, exp, counter persist
        // next op = ExpBase
        // ----------------------------------------------------
        let next_ebase: AB::Expr = next[SEL_EXP_BASE].into();

        builder
            .when_transition()
            .when(eacc.clone())
            .assert_one(next_ebase);

        for i in 0..FQ12_LIMBS {
            builder
                .when_transition()
                .when(eacc.clone())
                .assert_eq(
                    next[BASE_START + i],
                    local[BASE_START + i],
                );
        }

        for i in 0..EXP_LIMBS {
            builder
                .when_transition()
                .when(eacc.clone())
                .assert_eq(
                    next[EXP_START + i],
                    local[EXP_START + i],
                );
        }

        builder
            .when_transition()
            .when(eacc.clone())
            .assert_eq(
                next[EXP_COUNTER],
                local[EXP_COUNTER],
            );

        for i in 0..FQ12_LIMBS {
            let a: AB::Expr = local[ACC_START + i].into();
            let n: AB::Expr = next[ACC_START + i].into();

            builder
                .when_transition()
                .when(
                    eacc.clone()
                    * (AB::Expr::ONE - exp_bit.clone())
                )
                .assert_eq(n, a);
        }

        // ----------------------------------------------------
        // ExpBase:
        //
        // next.base = base^2 (generic mul)
        // next.acc  = acc
        // next.exp  = exp >> 1
        // next.counter = counter+1
        // next op is ExpAcc or Output.
        // ----------------------------------------------------
        for i in 0..FQ12_LIMBS {
            builder
                .when_transition()
                .when(ebase.clone())
                .assert_eq(
                    next[ACC_START + i],
                    local[ACC_START + i],
                );
        }

        builder
            .when_transition()
            .when(ebase.clone())
            .assert_eq(
                next[EXP_COUNTER],
                local[EXP_COUNTER] + AB::Expr::ONE,
            );

        let next_output: AB::Expr = next[SEL_OUTPUT].into();

        builder
            .when_transition()
            .when(ebase.clone())
            .assert_one(next_eacc + next_output);

        // Right shift exp exactly:
        //
        // local limb_i + B*carry_out
        //   = carry_in + 2*next limb_i
        //
        // carry_in = local exponent LSB for limb 0,
        // otherwise MSB(next limb_{i-1}).
        //
        // carry_out = MSB(next limb_i), except top = 0.
        for i in 0..EXP_LIMBS {
            let local_limb: AB::Expr =
                local[EXP_START + i].into();
            let next_limb: AB::Expr =
                next[EXP_START + i].into();

            let carry_in =
                if i == 0 {
                    exp_bit.clone()
                } else {
                    next[
                        EXP_BITS_START
                        + (i - 1) * LIMB_BITS
                        + (LIMB_BITS - 1)
                    ]
                    .into()
                };

            let carry_out: AB::Expr =
                if i + 1 == EXP_LIMBS {
                    AB::Expr::ZERO
                } else {
                    next[
                        EXP_BITS_START
                        + i * LIMB_BITS
                        + (LIMB_BITS - 1)
                    ]
                    .into()
                };

            builder
                .when_transition()
                .when(ebase.clone())
                .assert_eq(
                    local_limb
                    + carry_out * AB::Expr::from_u64(BASE),
                    carry_in
                    + next_limb.clone()
                    + next_limb,
                );
        }

        // ----------------------------------------------------
        // OUTPUT: exact exponent consumed, exact pairing value.
        // ----------------------------------------------------
        builder
            .when(output.clone())
            .assert_eq(
                local[EXP_COUNTER],
                AB::Expr::from_u64(EXP_BITS as u64),
            );

        for i in EXP_START..EXP_END {
            builder.when(output.clone()).assert_zero(local[i]);
        }

        for i in 0..FQ12_LIMBS {
            builder
                .when(output.clone())
                .assert_eq(
                    local[ACC_START + i],
                    AB::Expr::from_u64(
                        self.pairing_output_limbs[i]
                    ),
                );
        }

        let next_pad: AB::Expr = next[SEL_PAD].into();

        builder
            .when_transition()
            .when(output.clone())
            .assert_one(next_pad.clone());

        // PAD persists.
        builder
            .when_transition()
            .when(pad.clone())
            .assert_one(next_pad);

        for i in ACC_START..EXP_END {
            builder
                .when_transition()
                .when(pad.clone())
                .assert_eq(local[i], next[i]);
        }

        // ----------------------------------------------------
        // Generic multiplication operand binding
        // ----------------------------------------------------
        //
        // A,B are explicit columns. This keeps the modular
        // multiplication equation uniform across all phases.
        // ----------------------------------------------------

        // Miller square: A=acc, B=acc.
        for i in 0..FQ12_LIMBS {
            builder
                .when(msq.clone())
                .assert_eq(
                    local[OP_A_START + i],
                    local[ACC_START + i],
                );

            builder
                .when(msq.clone())
                .assert_eq(
                    local[OP_B_START + i],
                    local[ACC_START + i],
                );
        }

        // Miller line: A=acc; B is sparse(c0,c3,c4).
        for i in 0..FQ12_LIMBS {
            builder
                .when(mline.clone())
                .assert_eq(
                    local[OP_A_START + i],
                    local[ACC_START + i],
                );

            let coord = i / LIMBS;
            let limb = i % LIMBS;

            let sparse_b: AB::Expr =
                match coord {
                    0 => local[
                        LINE_START + 0 * LIMBS + limb
                    ].into(),
                    1 => local[
                        LINE_START + 1 * LIMBS + limb
                    ].into(),
                    6 => local[
                        LINE_START + 2 * LIMBS + limb
                    ].into(),
                    7 => local[
                        LINE_START + 3 * LIMBS + limb
                    ].into(),
                    8 => local[
                        LINE_START + 4 * LIMBS + limb
                    ].into(),
                    9 => local[
                        LINE_START + 5 * LIMBS + limb
                    ].into(),
                    _ => AB::Expr::ZERO,
                };

            builder
                .when(mline.clone())
                .assert_eq(
                    local[OP_B_START + i],
                    sparse_b,
                );
        }

        // ExpAcc: A=bit*acc, B=bit*base.
        for i in 0..FQ12_LIMBS {
            let a: AB::Expr = local[ACC_START + i].into();
            let b: AB::Expr = local[BASE_START + i].into();

            builder
                .when(eacc.clone())
                .assert_eq(
                    local[OP_A_START + i],
                    exp_bit.clone() * a,
                );

            builder
                .when(eacc.clone())
                .assert_eq(
                    local[OP_B_START + i],
                    exp_bit.clone() * b,
                );
        }

        // ExpBase: A=base, B=base.
        for i in 0..FQ12_LIMBS {
            builder
                .when(ebase.clone())
                .assert_eq(
                    local[OP_A_START + i],
                    local[BASE_START + i],
                );
            builder
                .when(ebase.clone())
                .assert_eq(
                    local[OP_B_START + i],
                    local[BASE_START + i],
                );
        }

        // ----------------------------------------------------
        // Multiplication witness bit constraints
        // ----------------------------------------------------
        let arith_active =
            msq.clone()
            + mline.clone()
            + ebase.clone()
            + eacc.clone() * exp_bit.clone();

        for r in 0..12 {
            let s: AB::Expr = local[Q_SIGN_START + r].into();

            builder
                .when(arith_active.clone())
                .assert_zero(
                    s.clone() * (s - AB::Expr::ONE)
                );

            for ql in 0..QW_LIMBS {
                let start =
                    Q_BITS_START
                    + (r * QW_LIMBS + ql) * LIMB_BITS;

                for b in 0..LIMB_BITS {
                    let bit: AB::Expr = local[start + b].into();

                    builder
                        .when(arith_active.clone())
                        .assert_zero(
                            bit.clone()
                            * (bit - AB::Expr::ONE)
                        );
                }
            }

            for c in 0..CARRY_COUNT {
                let sc =
                    CARRY_SIGN_START
                    + r * CARRY_COUNT + c;

                let sb: AB::Expr = local[sc].into();

                builder
                    .when(arith_active.clone())
                    .assert_zero(
                        sb.clone() * (sb - AB::Expr::ONE)
                    );

                let bs =
                    CARRY_BITS_START
                    + (r * CARRY_COUNT + c) * CARRY_BITS;

                for b in 0..CARRY_BITS {
                    let bit: AB::Expr = local[bs + b].into();

                    builder
                        .when(arith_active.clone())
                        .assert_zero(
                            bit.clone()
                            * (bit - AB::Expr::ONE)
                        );
                }
            }
        }

        // ----------------------------------------------------
        // Generic non-native Fq12 multiplication relation.
        // ----------------------------------------------------
        for r in 0..12 {
            let qsign: AB::Expr =
                local[Q_SIGN_START + r].into();
            let qfactor =
                AB::Expr::ONE - qsign.clone() - qsign;

            for t in 0..32 {
                let carry_cur =
                    if t == 0 {
                        AB::Expr::ZERO
                    } else {
                        carry_expr::<AB>(local, r, t - 1)
                    };

                let carry_next =
                    if t == 31 {
                        AB::Expr::ZERO
                    } else {
                        carry_expr::<AB>(local, r, t)
                    };

                let mut core = carry_cur;

                // Product coordinate limb t.
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

                            let a =
                                operand_limb::<AB>(
                                    local,
                                    OP_A_START,
                                    i,
                                    ai,
                                );

                            let b =
                                operand_limb::<AB>(
                                    local,
                                    OP_B_START,
                                    j,
                                    bj,
                                );

                            let term = a * b;
                            let mag = coeff.unsigned_abs();

                            let scaled =
                                term
                                * AB::Expr::from_u64(mag);

                            if coeff > 0 {
                                core = core + scaled;
                            } else {
                                core = core - scaled;
                            }
                        }
                    }
                }

                // -(k*q)_t
                let mut kq = AB::Expr::ZERO;

                for ki in 0..QW_LIMBS {
                    if t < ki {
                        continue;
                    }

                    let qj = t - ki;

                    if qj < LIMBS {
                        kq =
                            kq
                            + quotient_limb::<AB>(
                                local, r, ki
                            )
                            * AB::Expr::from_u64(
                                Q_LIMBS[qj]
                            );
                    }
                }

                core = core - qfactor.clone() * kq;

                core =
                    core
                    - carry_next
                        * AB::Expr::from_u64(BASE);

                // Target output differs only for ExpBase.
                if t < LIMBS {
                    let next_acc: AB::Expr =
                        next[
                            ACC_START + r * LIMBS + t
                        ]
                        .into();

                    let next_base: AB::Expr =
                        next[
                            BASE_START + r * LIMBS + t
                        ]
                        .into();

                    let acc_core =
                        core.clone() - next_acc;

                    let base_core =
                        core.clone() - next_base;

                    builder
                        .when_transition()
                        .when(msq.clone())
                        .assert_zero(acc_core.clone());

                    builder
                        .when_transition()
                        .when(mline.clone())
                        .assert_zero(acc_core.clone());

                    builder
                        .when_transition()
                        .when(eacc.clone() * exp_bit.clone())
                        .assert_zero(acc_core);

                    builder
                        .when_transition()
                        .when(ebase.clone())
                        .assert_zero(base_core);
                } else {
                    builder
                        .when_transition()
                        .when(msq.clone())
                        .assert_zero(core.clone());

                    builder
                        .when_transition()
                        .when(mline.clone())
                        .assert_zero(core.clone());

                    builder
                        .when_transition()
                        .when(eacc.clone() * exp_bit.clone())
                        .assert_zero(core.clone());

                    builder
                        .when_transition()
                        .when(ebase.clone())
                        .assert_zero(core);
                }
            }
        }

        // ----------------------------------------------------
        // ----------------------------------------------------
        // Fully integrated dynamic Miller-line + difference sub-AIRs.
        // ----------------------------------------------------
        let line_start = PAIR_WIDTH;
        let diff_start = PAIR_WIDTH + line_eval_layer::WIDTH;

        let line_local =
            &local[line_start..line_start + line_eval_layer::WIDTH];
        let diff_local =
            &local[diff_start..diff_start + diff_layer::WIDTH];
        let diff_next =
            &next[diff_start..diff_start + diff_layer::WIDTH];

        line_eval_layer::eval(
            builder,
            line_local,
            &local[..PAIR_WIDTH],
            diff_local,
            &public_values,
            &self.miller,
            mline.clone(),
        );

        diff_layer::eval(
            builder,
            diff_local,
            diff_next,
            &public_values[..COMMITMENT_POINTS * G1_POINT_LIMBS],
        );
    }
}

// ============================================================
// Generalized polynomial-equality helpers for this checkpoint
// ============================================================

pub type Polynomial = Vec<Fr>;

fn poly_eval(poly: &[Fr], x: Fr) -> Fr {
    let mut acc = Fr::zero();

    for coeff in poly.iter().rev() {
        acc *= x;
        acc += coeff;
    }

    acc
}

fn poly_trim(mut poly: Polynomial) -> Polynomial {
    while poly.len() > 1 && poly.last().is_some_and(|x| x.is_zero()) {
        poly.pop();
    }

    if poly.is_empty() {
        poly.push(Fr::zero());
    }

    poly
}

fn poly_sub(a: &[Fr], b: &[Fr]) -> Polynomial {
    let n = a.len().max(b.len());
    let mut out = vec![Fr::zero(); n];

    for i in 0..n {
        if i < a.len() {
            out[i] += a[i];
        }

        if i < b.len() {
            out[i] -= b[i];
        }
    }

    poly_trim(out)
}

fn poly_mul(a: &[Fr], b: &[Fr]) -> Polynomial {
    let mut out = vec![Fr::zero(); a.len() + b.len() - 1];

    for (i, ai) in a.iter().enumerate() {
        for (j, bj) in b.iter().enumerate() {
            out[i + j] += *ai * *bj;
        }
    }

    poly_trim(out)
}

fn add_scaled_poly(dst: &mut Polynomial, src: &[Fr], scale: Fr) {
    if dst.len() < src.len() {
        dst.resize(src.len(), Fr::zero());
    }

    for i in 0..src.len() {
        dst[i] += scale * src[i];
    }

    *dst = poly_trim(dst.clone());
}

fn divide_by_x_minus_z(delta: &[Fr], z: Fr) -> Polynomial {
    assert!(delta.len() >= 2);

    let n = delta.len() - 1;
    let mut q = vec![Fr::zero(); n];

    q[n - 1] = delta[n];

    if n >= 2 {
        for k in (1..n).rev() {
            q[k - 1] = delta[k] + z * q[k];
        }
    }

    let remainder = delta[0] + z * q[0];
    assert!(remainder.is_zero());

    poly_trim(q)
}

fn vanishing_poly(points: &[Fr]) -> Polynomial {
    let mut out = vec![Fr::from(1u64)];

    for z in points {
        out = poly_mul(
            &out,
            &[-*z, Fr::from(1u64)],
        );
    }

    out
}

// Self-contained demo setup helper.
//
// We evaluate at tau only while CONSTRUCTING the toy SRS-derived points.
// The STARK itself never receives tau.
fn commit_g1_at_tau(poly: &[Fr], tau: Fr) -> G1Affine {
    G1Affine::generator()
        .mul_bigint(poly_eval(poly, tau).into_bigint())
        .into_affine()
}

fn commit_g2_at_tau(poly: &[Fr], tau: Fr) -> G2Affine {
    G2Affine::generator()
        .mul_bigint(poly_eval(poly, tau).into_bigint())
        .into_affine()
}

// ============================================================
// Main
// ============================================================

pub fn prove_and_verify_three_equalities(
    input: &ThreeEqualityInput,
    run_negative_public_tests: bool,
) -> Result<IntegratedRunReport, String> {
    println!("============================================================");
    println!(" Plonky3 GENERALIZED BONEH FULLY-INTEGRATED ONE-PROOF");
    println!(" public commitments -> internal D_i -> dynamic Miller lines -> pairing product");
    println!("============================================================");
    println!();

    // -------------------------------------------------------------------------
    // Reusable three-statement input.
    // -------------------------------------------------------------------------
    let left_polys: [Polynomial; 3] = [
        input.statements[0].left.clone(),
        input.statements[1].left.clone(),
        input.statements[2].left.clone(),
    ];

    let right_polys: [Polynomial; 3] = [
        input.statements[0].right.clone(),
        input.statements[1].right.clone(),
        input.statements[2].right.clone(),
    ];

    let z = [
        input.statements[0].point,
        input.statements[1].point,
        input.statements[2].point,
    ];

    let gamma = input.gamma;

    // Experimental/demo KZG setup parameter.
    //
    // This API preserves the exact construction we have tested.  The next
    // production refactor should replace this scalar with explicit G1/G2 SRS
    // elements so the verifier never handles tau.
    let tau = input.demo_tau;

    // Fail cleanly before constructing quotient polynomials if the requested
    // semantic equality is false.
    for i in 0..3 {
        if poly_eval(&left_polys[i], z[i]) != poly_eval(&right_polys[i], z[i]) {
            return Err(format!(
                "statement {} is false: left(z) != right(z)",
                i + 1
            ));
        }
    }

    // -------------------------------------------------------------------------
    // Commit to both sides and form the three INTERNAL difference commitments.
    // -------------------------------------------------------------------------
    //
    // Public statement order:
    //
    //   C_f1,C_F1,C_f2,C_F2,C_f3,C_F3,W.
    //
    // D_1,D_2,D_3 are witness values only.
    //
    let mut commitments = Vec::<G1Affine>::with_capacity(COMMITMENT_POINTS);
    let mut d_vec = Vec::<G1Affine>::with_capacity(3);

    for i in 0..3 {
        debug_assert_eq!(
            poly_eval(&left_polys[i], z[i]),
            poly_eval(&right_polys[i], z[i]),
        );

        let c_left = commit_g1_at_tau(&left_polys[i], tau);
        let c_right = commit_g1_at_tau(&right_polys[i], tau);

        commitments.push(c_left);
        commitments.push(c_right);

        d_vec.push(
            (
                c_left.into_group()
                    - c_right.into_group()
            )
            .into_affine(),
        );
    }

    let commitment_points: [G1Affine; COMMITMENT_POINTS] =
        commitments.try_into().ok().unwrap();

    let d: [G1Affine; 3] =
        d_vec.try_into().ok().unwrap();

    // -------------------------------------------------------------------------
    // Construct q_i and aggregate:
    //
    //     h(X) = q_1(X) + gamma q_2(X) + gamma^2 q_3(X)
    //
    //     W = Commit(h).
    // -------------------------------------------------------------------------
    let mut h = vec![Fr::zero()];
    let mut gamma_i = Fr::from(1u64);

    for i in 0..3 {
        let delta = poly_sub(
            &left_polys[i],
            &right_polys[i],
        );

        let q_i = divide_by_x_minus_z(
            &delta,
            z[i],
        );

        add_scaled_poly(
            &mut h,
            &q_i,
            gamma_i,
        );

        gamma_i *= gamma;
    }

    let w = commit_g1_at_tau(&h, tau);

    // -------------------------------------------------------------------------
    // Encode the FINAL public statement.
    // -------------------------------------------------------------------------
    //
    // Public points:
    //
    //   C_f1,C_F1,C_f2,C_F2,C_f3,C_F3,W.
    //
    // Notice that D_i are absent.
    //
    let statement_points = [
        commitment_points[0],
        commitment_points[1],
        commitment_points[2],
        commitment_points[3],
        commitment_points[4],
        commitment_points[5],
        w,
    ];

    let mut statement_g1_limbs = [0u64; NUM_PUBLIC_VALUES];
    let mut public_values = Vec::with_capacity(NUM_PUBLIC_VALUES);

    for (p_index, point) in statement_points.iter().enumerate() {
        let x_limbs = fq_to_limbs(&point.x);
        let y_limbs = fq_to_limbs(&point.y);

        for l in 0..LIMBS {
            let idx = p_index * G1_POINT_LIMBS + l;
            statement_g1_limbs[idx] = x_limbs[l];
            public_values.push(Val::from_u64(x_limbs[l]));
        }

        for l in 0..LIMBS {
            let idx = p_index * G1_POINT_LIMBS + LIMBS + l;
            statement_g1_limbs[idx] = y_limbs[l];
            public_values.push(Val::from_u64(y_limbs[l]));
        }
    }

    // -------------------------------------------------------------------------
    // Construct the G2 vanishing-polynomial commitments.
    //
    // T = {z_1,z_2,z_3}
    //
    // Z_i(X) = product_{j != i}(X-z_j)
    // Z_T(X) = product_j(X-z_j)
    //
    // Q_i = [Z_i(tau)]G2
    // Q_T = [Z_T(tau)]G2
    //
    // In a real verifier these points come from the G2 SRS.  tau is not
    // required by the verifier.
    // -------------------------------------------------------------------------
    let z_t = vanishing_poly(&z);
    let q_t = commit_g2_at_tau(&z_t, tau);

    let mut q_side = Vec::<G2Affine>::with_capacity(3);

    for i in 0..3 {
        let mut others = Vec::<Fr>::with_capacity(2);

        for j in 0..3 {
            if i != j {
                others.push(z[j]);
            }
        }

        let z_i = vanishing_poly(&others);
        q_side.push(commit_g2_at_tau(&z_i, tau));
    }

    // -------------------------------------------------------------------------
    // Rewrite the aggregate verification equation as ONE product-to-identity.
    // -------------------------------------------------------------------------
    //
    // Start with:
    //
    //   e(D_1,Q_1)
    // * e([gamma]D_2,Q_2)
    // * e([gamma^2]D_3,Q_3)
    // * e(-W,Q_T)
    // = 1.
    //
    // By bilinearity we move the known demo challenge powers to G2:
    //
    //   e(D_1,Q_1)
    // * e(D_2,[gamma]Q_2)
    // * e(D_3,[gamma^2]Q_3)
    // * e(-W,Q_T)
    // = 1.
    //
    // This means the G1 pairing inputs are exactly the INTERNAL D_i values
    // proven by the appended commitment-difference sub-AIR.
    //
    let gamma_sq = gamma * gamma;

    let q2_weighted = q_side[1]
        .mul_bigint(gamma.into_bigint())
        .into_affine();

    let q3_weighted = q_side[2]
        .mul_bigint(gamma_sq.into_bigint())
        .into_affine();

    // Keep W itself on the G1 side so the dynamic Miller-line AIR can
    // bind pair 4 directly to the public W coordinates.  Move the minus
    // sign to G2 using bilinearity:
    //
    //     e(-W, Q_T) = e(W, -Q_T).
    //
    let neg_q_t = (-q_t.into_group()).into_affine();

    let pairs = [
        (d[0], q_side[0]),
        (d[1], q2_weighted),
        (d[2], q3_weighted),
        (w, neg_q_t),
    ];

    // Native reference check.
    let native_product = Bn254::multi_pairing(
        pairs.iter().map(|(p, _)| *p).collect::<Vec<_>>(),
        pairs.iter().map(|(_, q)| *q).collect::<Vec<_>>(),
    ).0;

    assert_eq!(
        native_product,
        Fq12::one(),
        "native generalized Boneh multi-pairing relation failed",
    );

    println!("Native generalized Boneh relation : PASS");
    println!("Pairing terms                     : 4");
    println!("Public G1 statement points        : 7 (6 commitments + W)");
    println!("Internal D_i points               : 3");
    println!("Miller strategy                   : shared product loop");
    println!("Final exponentiations             : 1");
    println!();

    // -------------------------------------------------------------------------
    // Build the STARK trace.
    // -------------------------------------------------------------------------
    let (
        trace,
        real_rows,
        height,
        miller_constants,
        pairing_product,
    ) = make_trace(
        &pairs,
        &commitment_points,
        &d,
    );

    assert_eq!(
        pairing_product,
        Fq12::one(),
        "STARK witness pairing product is not the identity",
    );

    // AIR output is fixed to the multiplicative identity in Fq12.
    let identity_coords = fq12_to_coords(&Fq12::one());
    let mut identity_limbs = [0u64; FQ12_LIMBS];

    for i in 0..12 {
        let l = fq_to_limbs(&identity_coords[i]);

        for j in 0..LIMBS {
            identity_limbs[i * LIMBS + j] = l[j];
        }
    }

    let final_exp_limbs =
        exp_to_limbs(&final_exponent_biguint());

    let air = PairingAir {
        miller: miller_constants,
        pairing_output_limbs: identity_limbs,
        final_exp_limbs,
        statement_g1_limbs,
    };

    let air_layout =
        AirLayout::from_air::<Val>(&air);

    let symbolic_constraints =
        get_symbolic_constraints::<Val, _>(
            &air,
            air_layout,
        );

    let num_air_constraints =
        symbolic_constraints.len();

    let max_constraint_degree =
        get_max_constraint_degree::<Val, _>(
            &air,
            air_layout,
            height,
        );

    println!("============================================================");
    println!(" MULTI-PAIRING STARK TRACE");
    println!("============================================================");
    println!();
    println!("Miller product steps       : {}", MILLER_STEPS);
    println!("Final-exp bit rounds       : {}", EXP_BITS);
    println!("Real rows                  : {}", real_rows);
    println!("Padded height              : {}", height);
    println!("Pairing subtrace width     : {}", PAIR_WIDTH);
    println!("Dynamic-line subtrace width: {}", line_eval_layer::WIDTH);
    println!("Difference subtrace width  : {}", diff_layer::WIDTH);
    println!("Combined trace width       : {}", PAIR_WIDTH + line_eval_layer::WIDTH + diff_layer::WIDTH);
    println!("Trace cells                : {}", (PAIR_WIDTH + line_eval_layer::WIDTH + diff_layer::WIDTH) * height);
    println!("AIR constraints            : {}", num_air_constraints);
    println!("Max constraint degree      : {}", max_constraint_degree);
    println!();

    // -------------------------------------------------------------------------
    // Standard Plonky3 configuration.
    // -------------------------------------------------------------------------
    type ByteHash = Sha256;
    type FieldHash = SerializingHasher<ByteHash>;

    let byte_hash = ByteHash {};
    let field_hash = FieldHash::new(byte_hash);

    type Compress =
        CompressionFunctionFromHasher<ByteHash, 2, 32>;

    let compress = Compress::new(byte_hash);

    type ValMmcs =
        MerkleTreeMmcs<Val, u8, FieldHash, Compress, 2, 32>;

    let val_mmcs =
        ValMmcs::new(field_hash, compress, 3);

    type ChallengeMmcs =
        ExtensionMmcs<Val, Challenge, ValMmcs>;

    let challenge_mmcs =
        ChallengeMmcs::new(val_mmcs.clone());

    type Dft = Radix2DitParallel<Val>;
    let dft = Dft::default();

    let fri_params =
        FriParameters::new_benchmark(challenge_mmcs);

    type Pcs =
        TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;

    let pcs =
        Pcs::new(dft, val_mmcs, fri_params);

    type Challenger =
        SerializingChallenger64<
            Val,
            HashChallenger<u8, ByteHash, 32>,
        >;

    let challenger =
        Challenger::from_hasher(vec![], byte_hash);

    type ConfigType =
        StarkConfig<Pcs, Challenge, Challenger>;

    let config = ConfigType::new(pcs, challenger);

    // -------------------------------------------------------------------------
    // Prove + verify.
    // -------------------------------------------------------------------------
    let prove_start = Instant::now();

    let proof =
        prove(
            &config,
            &air,
            trace,
            &public_values,
        )
        .expect(
            "generalized Boneh multi-pairing STARK generation failed",
        );

    let proving_time = prove_start.elapsed();

    let proof_bytes =
        postcard::to_allocvec(&proof)
            .expect("proof serialization failed");

    let verify_start = Instant::now();

    verify(
        &config,
        &air,
        &proof,
        &public_values,
    )
    .expect(
        "generalized Boneh multi-pairing STARK verification failed",
    );

    let verification_time = verify_start.elapsed();

    if run_negative_public_tests {
        // -------------------------------------------------------------------------
        // Negative public-statement matrix.
        // -------------------------------------------------------------------------
        //
        // Reuse the SAME valid proof and mutate the first x-limb of each public
        // G1 point in turn:
        //
        //   0 C_f1, 1 C_F1, 2 C_f2, 3 C_F2, 4 C_f3, 5 C_F3, 6 W.
        //
        // Every modified public statement must be rejected.
        let labels = [
            "C_f1", "C_F1", "C_f2", "C_F2", "C_f3", "C_F3", "W",
        ];

        for point_index in 0..PUBLIC_G1_POINTS {
            let mut wrong_values = public_values.clone();
            let start = point_index * G1_POINT_LIMBS;
            wrong_values[start] += Val::ONE;

            assert!(
                verify(
                    &config,
                    &air,
                    &proof,
                    &wrong_values,
                )
                .is_err(),
                "modified public {} unexpectedly verified",
                labels[point_index],
            );
        }
    }

    println!("============================================================");
    println!(" GENERALIZED BONEH MONOLITHIC STARK RESULT");
    println!("============================================================");
    println!();

    println!("Native pairing product = 1       : PASS");
    println!("Miller product in AIR            : PASS");
    println!("Single final exponentiation AIR  : PASS");
    println!("Final Fq12 output = 1 in AIR     : PASS");
    println!("Six public commitments bound     : PASS");
    println!("D_i = C_fi - C_Fi in same AIR    : PASS");
    println!("D_i absent from public input      : PASS");
    println!("Dynamic Miller lines use D_i/W   : PASS");
    println!("Public W bound                    : PASS");
    if run_negative_public_tests {
        println!("All 7 public G1 mutation tests    : PASS");
    }
    println!();
    println!("Proof count                      : 1");
    println!("Proof size                       : {} bytes", proof_bytes.len());
    println!("Proving time                     : {:?}", proving_time);
    println!("Verification time                : {:?}", verification_time);
    println!();

    println!("SUCCESS:");
    println!("One Plonky3 STARK proof now contains both:");
    println!("  (1) C_fi,C_Fi -> internal D_i, and");
    println!("  (2) the generalized four-term Boneh pairing-product check.");
    println!();
    println!("D_i are no longer public bridge values.");
    println!();
    println!("FULL INTEGRATION STATUS:");
    println!("The AIR derives evaluated Miller lines from raw prepared G2");
    println!("coefficients and the exact internal D_i/public W coordinates.");
    println!("No host-evaluated G1 Miller-line value is trusted by the AIR.");

    Ok(IntegratedRunReport {
        proof_size_bytes: proof_bytes.len(),
        proving_time,
        verification_time,
        trace_height: height,
        trace_width: PAIR_WIDTH + line_eval_layer::WIDTH + diff_layer::WIDTH,
        trace_cells: (PAIR_WIDTH + line_eval_layer::WIDTH + diff_layer::WIDTH) * height,
        air_constraints: num_air_constraints,
        max_constraint_degree,
        negative_public_tests_ran: run_negative_public_tests,
    })
}

// ============================================================
// Reusable public API types
// ============================================================

#[derive(Clone)]
pub struct EqualityStatement {
    /// Polynomial coefficients in ascending order:
    /// [a_0, a_1, ...] represents a_0 + a_1 X + ...
    pub left: Polynomial,
    pub right: Polynomial,
    pub point: Fr,
}

#[derive(Clone)]
pub struct ThreeEqualityInput {
    pub statements: [EqualityStatement; 3],
    pub gamma: Fr,

    /// Experimental/demo setup scalar used to synthesize KZG SRS points.
    /// Keep this only for the current research prototype.
    pub demo_tau: Fr,
}

#[derive(Clone, Debug)]
pub struct IntegratedRunReport {
    pub proof_size_bytes: usize,
    pub proving_time: std::time::Duration,
    pub verification_time: std::time::Duration,
    pub trace_height: usize,
    pub trace_width: usize,
    pub trace_cells: usize,
    pub air_constraints: usize,
    pub max_constraint_degree: usize,
    pub negative_public_tests_ran: bool,
}

pub fn demo_three_equality_input() -> ThreeEqualityInput {
    ThreeEqualityInput {
        statements: [
            EqualityStatement {
                left: vec![Fr::from(3u64), Fr::from(1u64)],
                right: vec![Fr::from(5u64)],
                point: Fr::from(2u64),
            },
            EqualityStatement {
                left: vec![Fr::from(2u64), Fr::zero(), Fr::from(1u64)],
                right: vec![Fr::from(5u64), Fr::from(2u64)],
                point: Fr::from(3u64),
            },
            EqualityStatement {
                left: vec![Fr::from(1u64), Fr::from(2u64)],
                right: vec![Fr::from(5u64), Fr::from(1u64)],
                point: Fr::from(4u64),
            },
        ],
        gamma: Fr::from(2u64),
        demo_tau: Fr::from(11u64),
    }
}

fn main() {
    let input = demo_three_equality_input();

    let report = prove_and_verify_three_equalities(
        &input,
        true,
    )
    .expect("fully integrated three-equality proof failed");

    println!();
    println!("REUSABLE API REPORT");
    println!("-------------------");
    println!("proof bytes         : {}", report.proof_size_bytes);
    println!("trace width         : {}", report.trace_width);
    println!("trace height        : {}", report.trace_height);
    println!("AIR constraints     : {}", report.air_constraints);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polynomial_evaluation_matches_demo_equalities() {
        let input = demo_three_equality_input();

        for s in &input.statements {
            assert_eq!(
                poly_eval(&s.left, s.point),
                poly_eval(&s.right, s.point),
            );
        }
    }

    #[test]
    fn quotient_reconstructs_difference() {
        let input = demo_three_equality_input();

        for s in &input.statements {
            let delta = poly_sub(&s.left, &s.right);
            let q = divide_by_x_minus_z(&delta, s.point);

            let reconstructed = poly_mul(
                &q,
                &[-s.point, Fr::from(1u64)],
            );

            assert_eq!(
                poly_trim(reconstructed),
                poly_trim(delta),
            );
        }
    }

    #[test]
    fn vanishing_polynomial_is_zero_on_all_points() {
        let input = demo_three_equality_input();
        let points = input.statements.map(|s| s.point);
        let z_t = vanishing_poly(&points);

        for point in points {
            assert!(poly_eval(&z_t, point).is_zero());
        }
    }

    #[test]
    fn false_semantic_statement_is_rejected_before_proving() {
        let mut input = demo_three_equality_input();

        // Change f_2(X) by +1 at the constant coefficient.
        input.statements[1].left[0] += Fr::from(1u64);

        let result = prove_and_verify_three_equalities(
            &input,
            false,
        );

        assert!(result.is_err());
    }

    /// Full end-to-end STARK test.
    ///
    /// This is ignored by default because the current prototype takes about
    /// a minute to prove and allocates a ~460M-cell trace.
    ///
    /// Run explicitly with:
    ///
    /// `cargo test --release -p p3-uni-stark --example \
    /// bn254_boneh_integrated_api full_integrated_proof_and_public_tamper_tests \
    /// -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn full_integrated_proof_and_public_tamper_tests() {
        let input = demo_three_equality_input();

        let report = prove_and_verify_three_equalities(
            &input,
            true,
        )
        .expect("valid fully-integrated instance must verify");

        assert!(report.proof_size_bytes > 0);
        assert_eq!(report.trace_height, 8192);
        assert_eq!(report.trace_width, 56098);
        assert!(report.negative_public_tests_ran);
    }
}
