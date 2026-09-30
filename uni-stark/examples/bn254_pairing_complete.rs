
use std::time::Instant;

use ark_bn254::{Bn254, Config, Fq, Fq2, Fq6, Fq12, Fr, G1Affine, G2Affine};
use ark_ec::{
    bn::{BnConfig, G2Prepared},
    pairing::Pairing,
    AffineRepr,
};
use ark_ff::{BigInteger, Field, One, PrimeField, Zero};
use num_bigint::{BigInt, BigUint, Sign};

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
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
const BASE: u64 = 1 << LIMB_BITS;

const FQ12_COORDS: usize = 12;
const FQ12_LIMBS: usize = FQ12_COORDS * LIMBS;

const Q_LIMBS: [u64; LIMBS] = [
    64839, 55420, 35862, 15392,
    51853, 26737, 27281, 38785,
    22621, 33153, 17846, 47184,
    41001, 57649, 20082, 12388,
];

const MILLER_STEPS: usize = 150;

// (q^12 - 1)/r has 2790 bits for BN254.
const EXP_BITS: usize = 2790;
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
const WIDTH: usize =
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
    line_limbs: [u64; LINE_LIMBS],
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
    let q = modulus_biguint();
    let r = scalar_modulus_biguint();
    (q.pow(12u32) - BigUint::from(1u64)) / r
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

// ============================================================
// Build exact fixed Miller program for e(G1,G2)
// ============================================================

fn build_miller_program(
) -> (Vec<NativeRow>, Vec<MillerStepConst>, Fq12) {
    let p = G1Affine::generator();
    let q = G2Affine::generator();
    let prepared = G2Prepared::<Config>::from(q);

    let mut rows = Vec::<NativeRow>::new();
    let mut constants = Vec::<MillerStepConst>::new();

    let mut f = Fq12::one();
    let mut coeff_index = 0usize;
    let mut step = 0usize;

    for i in (1..Config::ATE_LOOP_COUNT.len()).rev() {
        let digit = Config::ATE_LOOP_COUNT[i - 1];

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
                line_limbs: [0u64; LINE_LIMBS],
            });

            let direct = fq12_mul_direct(&f, &f);
            let mut ark = f;
            ark.square_in_place();
            assert_eq!(direct, ark);

            f = ark;
            step += 1;
        }

        let coeff = prepared.ell_coeffs[coeff_index];
        let line = evaluated_line(&coeff, &p);

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
            line_limbs: line_to_limbs(&line),
        });

        let sparse = sparse_line_fq12(&line);
        let direct = fq12_mul_direct(&f, &sparse);

        f.mul_by_034(&line.0, &line.1, &line.2);

        assert_eq!(
            direct, f,
            "sparse direct Fq12 multiplication != arkworks mul_by_034"
        );

        coeff_index += 1;
        step += 1;

        if digit == 1 || digit == -1 {
            let coeff = prepared.ell_coeffs[coeff_index];
            let line = evaluated_line(&coeff, &p);

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
                line_limbs: line_to_limbs(&line),
            });

            let sparse = sparse_line_fq12(&line);
            let direct = fq12_mul_direct(&f, &sparse);

            f.mul_by_034(&line.0, &line.1, &line.2);

            assert_eq!(direct, f);

            coeff_index += 1;
            step += 1;
        }
    }

    assert!(
        !Config::X_IS_NEGATIVE,
        "this first complete pairing trace assumes BN254 X_IS_NEGATIVE=false"
    );

    for _ in 0..2 {
        let coeff = prepared.ell_coeffs[coeff_index];
        let line = evaluated_line(&coeff, &p);

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
            line_limbs: line_to_limbs(&line),
        });

        let sparse = sparse_line_fq12(&line);
        let direct = fq12_mul_direct(&f, &sparse);

        f.mul_by_034(&line.0, &line.1, &line.2);
        assert_eq!(direct, f);

        coeff_index += 1;
        step += 1;
    }

    assert_eq!(coeff_index, prepared.ell_coeffs.len());
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
) -> (Vec<NativeRow>, Vec<MillerStepConst>, Fq12) {
    let (mut rows, miller_constants, miller_out) =
        build_miller_program();

    let final_exp = final_exponent_biguint();
    assert_eq!(final_exp.bits() as usize, EXP_BITS);

    // Transition row: Miller output -> (acc=1, base=Miller, exp=E).
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

    let direct_pairing = Bn254::pairing(
        G1Affine::generator(),
        G2Affine::generator(),
    ).0;

    assert_eq!(
        acc, direct_pairing,
        "binary final exponentiation result != arkworks pairing"
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

    (rows, miller_constants, direct_pairing)
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
    let mut row = Vec::with_capacity(WIDTH);

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
        for _ in Q_SIGN_START..WIDTH {
            row.push(Val::ZERO);
        }
    }

    assert_eq!(row.len(), WIDTH);
    row
}

// ============================================================
// Trace construction
// ============================================================

fn make_trace(
) -> (
    RowMajorMatrix<Val>,
    usize,
    usize,
    Vec<MillerStepConst>,
    Fq12,
) {
    let (schedule, constants, pairing_out) =
        build_full_schedule();

    let real_rows = schedule.len();
    let height = real_rows.next_power_of_two();

    let mut values = Vec::with_capacity(WIDTH * height);

    for i in 0..schedule.len() {
        let next =
            if i + 1 < schedule.len() {
                &schedule[i + 1]
            } else {
                &schedule[i]
            };

        values.extend_from_slice(
            &encode_row(&schedule[i], next)
        );
    }

    let terminal = schedule.last().unwrap().clone();

    while values.len() / WIDTH < height {
        values.extend_from_slice(
            &encode_row(&terminal, &terminal)
        );
    }

    (
        RowMajorMatrix::new(values, WIDTH),
        real_rows,
        height,
        constants,
        pairing_out,
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

// ============================================================
// AIR
// ============================================================

struct PairingAir {
    miller: Vec<MillerStepConst>,
    pairing_output_limbs: [u64; FQ12_LIMBS],
    final_exp_limbs: [u64; EXP_LIMBS],
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

        // Bind every line limb to the exact BN254 prepared line
        // selected by the Miller step identity.
        for ll in 0..LINE_LIMBS {
            let mut expected = AB::Expr::ZERO;

            for j in 0..MILLER_STEPS {
                let c = self.miller[j].line_limbs[ll];
                if c != 0 {
                    let id: AB::Expr =
                        local[MILLER_ID_START + j].into();
                    expected =
                        expected
                        + id * AB::Expr::from_u64(c);
                }
            }

            builder.assert_eq(
                local[LINE_START + ll],
                expected,
            );
        }

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
    }
}

// ============================================================
// Main
// ============================================================

fn main() {
    println!("==============================================");
    println!(" Plonky3 BN254 COMPLETE PAIRING");
    println!(" Miller loop + final exponentiation");
    println!("==============================================");
    println!();

    println!("Native proving field : Goldilocks");
    println!("Pairing curve        : BN254");
    println!("Pairing input        : e(G1 generator, G2 generator)");
    println!("Fq representation    : 16 x 16-bit limbs");
    println!("Final exponent bits  : {}", EXP_BITS);
    println!();

    let (
        trace,
        real_rows,
        height,
        miller_constants,
        pairing_output,
    ) = make_trace();

    let pairing_coords = fq12_to_coords(&pairing_output);
    let mut pairing_output_limbs = [0u64; FQ12_LIMBS];

    for i in 0..12 {
        let l = fq_to_limbs(&pairing_coords[i]);
        for j in 0..LIMBS {
            pairing_output_limbs[i * LIMBS + j] = l[j];
        }
    }

    let final_exp_limbs =
        exp_to_limbs(&final_exponent_biguint());

    let air = PairingAir {
        miller: miller_constants,
        pairing_output_limbs,
        final_exp_limbs,
    };

    println!("==============================================");
    println!(" Trace layout");
    println!("==============================================");
    println!();

    println!("Miller steps         : {}", MILLER_STEPS);
    println!("Final-exp bit rounds : {}", EXP_BITS);
    println!("Real rows            : {}", real_rows);
    println!("Padded height        : {}", height);
    println!("Trace width          : {}", WIDTH);
    println!("Trace cells          : {}", WIDTH * height);
    println!();

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

    let prove_start = Instant::now();

    let proof =
        prove(
            &config,
            &air,
            trace,
            &[],
        )
        .expect("proof generation failed");

    let proving_time = prove_start.elapsed();

    let proof_bytes =
        postcard::to_allocvec(&proof)
            .expect("proof serialization failed");

    let verify_start = Instant::now();

    verify(
        &config,
        &air,
        &proof,
        &[],
    )
    .expect("verification failed");

    let verification_time = verify_start.elapsed();

    println!("==============================================");
    println!(" COMPLETE PAIRING STARK RESULT");
    println!("==============================================");
    println!();

    println!("Proof size        : {} bytes", proof_bytes.len());
    println!("Proving time      : {:?}", proving_time);
    println!("Verification time : {:?}", verification_time);
    println!();

    println!("SUCCESS:");
    println!("Plonky3 proved the complete fixed BN254 pairing");
    println!("e(G1 generator, G2 generator), including:");
    println!("  - exact Miller operation schedule");
    println!("  - exact 87 prepared Miller lines");
    println!("  - every Fq12 square/multiplication");
    println!("  - the full 2790-bit final exponentiation");
    println!("  - equality to arkworks BN254 pairing output");
}
