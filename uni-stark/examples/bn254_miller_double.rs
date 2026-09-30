use std::time::Instant;

use num_bigint::BigUint;

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
const WIDE_LIMBS: usize = 32;
const LIMB_BITS: usize = 16;
const CARRY_BITS: usize = 20;

const RADIX: u64 = 1 << LIMB_BITS;

type Fq = [u64; LIMBS];
type Fq2 = (Fq, Fq);

// ============================================================
// BN254 Fq modulus
// ============================================================

const Q: Fq = [
    64839, 55420, 35862, 15392,
    51853, 26737, 27281, 38785,
    22621, 33153, 17846, 47184,
    41001, 57649, 20082, 12388,
];

// ============================================================
// Standard BN254 G2 generator.
//
// X = X0 + X1*u
// Y = Y0 + Y1*u
// ============================================================

const G2_X0: Fq = [
    63213, 55698, 48476, 18142,
    56029, 63326, 8916, 26435,
    17529, 24156, 102, 17002,
    7798, 4639, 57071, 6144,
];

const G2_X1: Fq = [
    4802, 44787, 34231, 38884,
    59154, 13737, 18739, 61866,
    23845, 12795, 49079, 29280,
    18490, 37389, 37779, 6542,
];

const G2_Y0: Fq = [
    32170, 26362, 52225, 19686,
    54139, 3139, 59241, 58321,
    16527, 36299, 29056, 19115,
    28139, 56204, 24229, 4808,
];

const G2_Y1: Fq = [
    38747, 53538, 56028, 21932,
    36595, 28851, 12595, 48203,
    13205, 26892, 39341, 60574,
    61557, 22623, 35280, 2310,
];

// ============================================================
// For Z = 1:
//
// E = B2 * (3*C)
//   = B2 * 3
//
// This is a fixed curve constant for our first line checkpoint.
// ============================================================

const E0: Fq = [
    45089, 48361, 40038, 7926,
    14799, 15536, 31885, 61979,
    36751, 20638, 48734, 5273,
    49080, 43462, 15068, 8309,
];

const E1: Fq = [
    16758, 37193, 14099, 44520,
    18680, 44935, 57742, 62958,
    63965, 52362, 3849, 26502,
    64638, 45266, 15120, 453,
];

const ZERO_FQ: Fq = [0; LIMBS];

const ONE_FQ: Fq = [
    1, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
];

// ============================================================
// One Fq modular multiplication block
//
// a*b = c + k*q
// ============================================================

const A_OFF: usize = 0;
const B_OFF: usize = A_OFF + LIMBS;
const C_OFF: usize = B_OFF + LIMBS;
const K_OFF: usize = C_OFF + LIMBS;

const AB_OFF: usize = K_OFF + LIMBS;

const AB_CARRY_OFF: usize =
    AB_OFF + WIDE_LIMBS;

const KQ_OFF: usize =
    AB_CARRY_OFF + 33;

const KQ_CARRY_OFF: usize =
    KQ_OFF + WIDE_LIMBS;

const ADD_CARRY_OFF: usize =
    KQ_CARRY_OFF + 33;

const RANGE_LIMBS: usize =
    4 * LIMBS + 2 * WIDE_LIMBS;

const LIMB_BIT_OFF: usize =
    ADD_CARRY_OFF + 33;

const LIMB_BIT_COLS: usize =
    RANGE_LIMBS * LIMB_BITS;

const MUL_CARRIES: usize = 66;

const CARRY_BIT_OFF: usize =
    LIMB_BIT_OFF + LIMB_BIT_COLS;

const CARRY_BIT_COLS: usize =
    MUL_CARRIES * CARRY_BITS;

const MODMUL_WIDTH: usize =
    CARRY_BIT_OFF + CARRY_BIT_COLS;

// ============================================================
// Generic modular-add relation:
//
// a+b = c+r*q
// ============================================================

const MODADD_CARRY_COUNT: usize = 17;

const MODADD_WIDTH: usize =
    1
    + MODADD_CARRY_COUNT
    + MODADD_CARRY_COUNT;

// ============================================================
// One Fq2-square block
//
// 3 Fq multiplications:
//
// M1 = a0^2
// M2 = a1^2
// M3 = a0*a1
//
// output:
//
// c0 = a0^2-a1^2
// c1 = 2*a0*a1
// ============================================================

const SQ_MUL1: usize = 0;

const SQ_MUL2: usize =
    SQ_MUL1 + MODMUL_WIDTH;

const SQ_MUL3: usize =
    SQ_MUL2 + MODMUL_WIDTH;

const SQ_C0: usize =
    SQ_MUL3 + MODMUL_WIDTH;

const SQ_C1: usize =
    SQ_C0 + LIMBS;

const SQ_MODADD_START: usize =
    SQ_C1 + LIMBS;

const SQ_NUM_MODADDS: usize = 2;

const SQ_OUTPUT_BITS: usize =
    SQ_MODADD_START
    + SQ_NUM_MODADDS * MODADD_WIDTH;

const SQ_EXTRA_LIMBS: usize =
    2 * LIMBS;

const SQ_WIDTH: usize =
    SQ_OUTPUT_BITS
    + SQ_EXTRA_LIMBS * LIMB_BITS;

// ============================================================
// Three Fq2 squares:
//
// 1. B  = Y^2
// 2. S2 = (Y+Z)^2
// 3. J  = X^2
// ============================================================

const SQ_B_BASE: usize = 0;

const SQ_S2_BASE: usize =
    SQ_B_BASE + SQ_WIDTH;

const SQ_J_BASE: usize =
    SQ_S2_BASE + SQ_WIDTH;

const SQUARES_END: usize =
    SQ_J_BASE + SQ_WIDTH;

// ============================================================
// Extra Fq2 witness values
//
// Each occupies 32 columns.
//
// Z       = 1
// S       = Y+Z
// BC      = B+C = B+1
// H       = S2-BC
// I       = E-B
// 2J
// 3J
// -H
// E       = fixed
// ZERO
// ============================================================

const FQ2_COLS: usize = 2 * LIMBS;

const Z_START: usize =
    SQUARES_END;

const S_START: usize =
    Z_START + FQ2_COLS;

const BC_START: usize =
    S_START + FQ2_COLS;

const H_START: usize =
    BC_START + FQ2_COLS;

const I_START: usize =
    H_START + FQ2_COLS;

const TWOJ_START: usize =
    I_START + FQ2_COLS;

const THREEJ_START: usize =
    TWOJ_START + FQ2_COLS;

const NEG_H_START: usize =
    THREEJ_START + FQ2_COLS;

const E_START: usize =
    NEG_H_START + FQ2_COLS;

const ZERO_START: usize =
    E_START + FQ2_COLS;

const EXTRA_END: usize =
    ZERO_START + FQ2_COLS;

const EXTRA_FQ2_VALUES: usize = 10;

const EXTRA_LIMBS: usize =
    EXTRA_FQ2_VALUES * FQ2_COLS;

// ============================================================
// Outer Fq modular additions.
//
// 7 Fq2 equations = 14 Fq equations.
//
// 0/1   Y + Z = S
// 2/3   B + Z = BC
// 4/5   H + BC = S2
// 6/7   I + B = E
// 8/9   J + J = 2J
// 10/11 2J + J = 3J
// 12/13 -H + H = 0
// ============================================================

const OUTER_MODADDS: usize = 14;

const OUTER_MODADD_START: usize =
    EXTRA_END;

const EXTRA_BITS_START: usize =
    OUTER_MODADD_START
    + OUTER_MODADDS * MODADD_WIDTH;

const WIDTH: usize =
    EXTRA_BITS_START
    + EXTRA_LIMBS * LIMB_BITS;

// ============================================================
// Small index helpers
// ============================================================

fn fq2_c0(start: usize) -> usize {
    start
}

fn fq2_c1(start: usize) -> usize {
    start + LIMBS
}

fn sq_out0(base: usize) -> usize {
    base + SQ_C0
}

fn sq_out1(base: usize) -> usize {
    base + SQ_C1
}

// ============================================================
// Fq modular multiplication AIR
// ============================================================

fn eval_modmul<AB: AirBuilder>(
    builder: &mut AB,
    row: &[AB::Var],
    base: usize,
) {
    let radix =
        AB::Expr::from_u64(RADIX);

    let a_start = base + A_OFF;
    let b_start = base + B_OFF;
    let c_start = base + C_OFF;
    let k_start = base + K_OFF;

    let ab_start =
        base + AB_OFF;

    let ab_carry_start =
        base + AB_CARRY_OFF;

    let kq_start =
        base + KQ_OFF;

    let kq_carry_start =
        base + KQ_CARRY_OFF;

    let add_carry_start =
        base + ADD_CARRY_OFF;

    let limb_bit_start =
        base + LIMB_BIT_OFF;

    let carry_bit_start =
        base + CARRY_BIT_OFF;

    // --------------------------------------------------------
    // a*b
    // --------------------------------------------------------

    let carry0: AB::Expr =
        row[ab_carry_start].into();

    builder.assert_zero(carry0);

    for out in 0..WIDE_LIMBS {
        let mut acc: AB::Expr =
            row[
                ab_carry_start + out
            ]
            .into();

        for i in 0..LIMBS {
            if out >= i {
                let j = out - i;

                if j < LIMBS {
                    let a_i: AB::Expr =
                        row[
                            a_start + i
                        ]
                        .into();

                    let b_j: AB::Expr =
                        row[
                            b_start + j
                        ]
                        .into();

                    acc =
                        acc + a_i * b_j;
                }
            }
        }

        let result: AB::Expr =
            row[
                ab_start + out
            ]
            .into();

        let carry_out: AB::Expr =
            row[
                ab_carry_start
                + out
                + 1
            ]
            .into();

        builder.assert_zero(
            acc
                - result
                - radix.clone()
                    * carry_out
        );
    }

    let final_ab: AB::Expr =
        row[
            ab_carry_start
            + WIDE_LIMBS
        ]
        .into();

    builder.assert_zero(final_ab);

    // --------------------------------------------------------
    // k*q
    // --------------------------------------------------------

    let kq0: AB::Expr =
        row[kq_carry_start].into();

    builder.assert_zero(kq0);

    for out in 0..WIDE_LIMBS {
        let mut acc: AB::Expr =
            row[
                kq_carry_start + out
            ]
            .into();

        for i in 0..LIMBS {
            if out >= i {
                let j = out - i;

                if j < LIMBS {
                    let k_i: AB::Expr =
                        row[
                            k_start + i
                        ]
                        .into();

                    let q_j =
                        AB::Expr::from_u64(
                            Q[j]
                        );

                    acc =
                        acc + k_i * q_j;
                }
            }
        }

        let result: AB::Expr =
            row[
                kq_start + out
            ]
            .into();

        let carry_out: AB::Expr =
            row[
                kq_carry_start
                + out
                + 1
            ]
            .into();

        builder.assert_zero(
            acc
                - result
                - radix.clone()
                    * carry_out
        );
    }

    let final_kq: AB::Expr =
        row[
            kq_carry_start
            + WIDE_LIMBS
        ]
        .into();

    builder.assert_zero(final_kq);

    // --------------------------------------------------------
    // c + k*q = a*b
    // --------------------------------------------------------

    let add0: AB::Expr =
        row[add_carry_start].into();

    builder.assert_zero(add0);

    for i in 0..WIDE_LIMBS {
        let carry_in: AB::Expr =
            row[
                add_carry_start + i
            ]
            .into();

        let carry_out: AB::Expr =
            row[
                add_carry_start
                + i
                + 1
            ]
            .into();

        let kq_i: AB::Expr =
            row[
                kq_start + i
            ]
            .into();

        let ab_i: AB::Expr =
            row[
                ab_start + i
            ]
            .into();

        let c_i =
            if i < LIMBS {
                let x: AB::Expr =
                    row[
                        c_start + i
                    ]
                    .into();

                x
            } else {
                AB::Expr::ZERO
            };

        builder.assert_zero(
            c_i
                + kq_i
                + carry_in
                - ab_i
                - radix.clone()
                    * carry_out.clone()
        );

        builder.assert_zero(
            carry_out.clone()
                * (
                    carry_out
                    - AB::Expr::ONE
                )
        );
    }

    let final_add: AB::Expr =
        row[
            add_carry_start
            + WIDE_LIMBS
        ]
        .into();

    builder.assert_zero(final_add);

    // --------------------------------------------------------
    // 16-bit range checks
    // --------------------------------------------------------

    for range_idx in 0..RANGE_LIMBS {
        let value_col =
            if range_idx < LIMBS {
                a_start + range_idx
            } else if range_idx < 2 * LIMBS {
                b_start
                    + range_idx
                    - LIMBS
            } else if range_idx < 3 * LIMBS {
                c_start
                    + range_idx
                    - 2 * LIMBS
            } else if range_idx < 4 * LIMBS {
                k_start
                    + range_idx
                    - 3 * LIMBS
            } else if range_idx
                < 4 * LIMBS + WIDE_LIMBS
            {
                ab_start
                    + range_idx
                    - 4 * LIMBS
            } else {
                kq_start
                    + range_idx
                    - 4 * LIMBS
                    - WIDE_LIMBS
            };

        let value: AB::Expr =
            row[value_col].into();

        let mut reconstructed =
            AB::Expr::ZERO;

        let mut power = 1u64;

        for bit in 0..LIMB_BITS {
            let col =
                limb_bit_start
                + range_idx * LIMB_BITS
                + bit;

            let b: AB::Expr =
                row[col].into();

            builder.assert_zero(
                b.clone()
                    * (
                        b.clone()
                        - AB::Expr::ONE
                    )
            );

            reconstructed =
                reconstructed
                + AB::Expr::from_u64(
                    power
                ) * b;

            power <<= 1;
        }

        builder.assert_zero(
            value - reconstructed
        );
    }

    // --------------------------------------------------------
    // carry range checks
    // --------------------------------------------------------

    for idx in 0..MUL_CARRIES {
        let value_col =
            if idx < 33 {
                ab_carry_start + idx
            } else {
                kq_carry_start
                    + idx
                    - 33
            };

        let value: AB::Expr =
            row[value_col].into();

        let mut reconstructed =
            AB::Expr::ZERO;

        let mut power = 1u64;

        for bit in 0..CARRY_BITS {
            let col =
                carry_bit_start
                + idx * CARRY_BITS
                + bit;

            let b: AB::Expr =
                row[col].into();

            builder.assert_zero(
                b.clone()
                    * (
                        b.clone()
                        - AB::Expr::ONE
                    )
            );

            reconstructed =
                reconstructed
                + AB::Expr::from_u64(
                    power
                ) * b;

            power <<= 1;
        }

        builder.assert_zero(
            value - reconstructed
        );
    }
}

// ============================================================
// Generic modular-add AIR:
//
// a+b = c+r*q
// ============================================================

fn eval_modadd_at<AB: AirBuilder>(
    builder: &mut AB,
    row: &[AB::Var],
    rel_base: usize,
    a_start: usize,
    b_start: usize,
    c_start: usize,
) {
    let r_col =
        rel_base;

    let lhs_carry =
        rel_base + 1;

    let rhs_carry =
        lhs_carry
        + MODADD_CARRY_COUNT;

    let r: AB::Expr =
        row[r_col].into();

    builder.assert_zero(
        r.clone()
            * (
                r.clone()
                - AB::Expr::ONE
            )
    );

    let lhs0: AB::Expr =
        row[lhs_carry].into();

    let rhs0: AB::Expr =
        row[rhs_carry].into();

    builder.assert_zero(lhs0);
    builder.assert_zero(rhs0);

    let radix =
        AB::Expr::from_u64(RADIX);

    for i in 0..LIMBS {
        let a: AB::Expr =
            row[
                a_start + i
            ]
            .into();

        let b: AB::Expr =
            row[
                b_start + i
            ]
            .into();

        let c: AB::Expr =
            row[
                c_start + i
            ]
            .into();

        let lhs_in: AB::Expr =
            row[
                lhs_carry + i
            ]
            .into();

        let lhs_out: AB::Expr =
            row[
                lhs_carry
                + i
                + 1
            ]
            .into();

        let rhs_in: AB::Expr =
            row[
                rhs_carry + i
            ]
            .into();

        let rhs_out: AB::Expr =
            row[
                rhs_carry
                + i
                + 1
            ]
            .into();

        builder.assert_zero(
            lhs_out.clone()
                * (
                    lhs_out.clone()
                    - AB::Expr::ONE
                )
        );

        builder.assert_zero(
            rhs_out.clone()
                * (
                    rhs_out.clone()
                    - AB::Expr::ONE
                )
        );

        let lhs_digit =
            a
            + b
            + lhs_in
            - radix.clone()
                * lhs_out;

        let rhs_digit =
            c
            + r.clone()
                * AB::Expr::from_u64(
                    Q[i]
                )
            + rhs_in
            - radix.clone()
                * rhs_out;

        builder.assert_zero(
            lhs_digit
                - rhs_digit
        );
    }

    let lhs_final: AB::Expr =
        row[
            lhs_carry + LIMBS
        ]
        .into();

    let rhs_final: AB::Expr =
        row[
            rhs_carry + LIMBS
        ]
        .into();

    builder.assert_zero(
        lhs_final
            - rhs_final
    );
}

// ============================================================
// Fq2-square AIR
// ============================================================

fn eval_fq2_square<AB: AirBuilder>(
    builder: &mut AB,
    row: &[AB::Var],
    base: usize,
) {
    eval_modmul(
        builder,
        row,
        base + SQ_MUL1,
    );

    eval_modmul(
        builder,
        row,
        base + SQ_MUL2,
    );

    eval_modmul(
        builder,
        row,
        base + SQ_MUL3,
    );

    // M1 must be a0*a0
    // M2 must be a1*a1
    // M3 must be a0*a1

    for i in 0..LIMBS {
        let m1a: AB::Expr =
            row[
                base
                + SQ_MUL1
                + A_OFF
                + i
            ]
            .into();

        let m1b: AB::Expr =
            row[
                base
                + SQ_MUL1
                + B_OFF
                + i
            ]
            .into();

        builder.assert_zero(
            m1a.clone() - m1b
        );

        let m2a: AB::Expr =
            row[
                base
                + SQ_MUL2
                + A_OFF
                + i
            ]
            .into();

        let m2b: AB::Expr =
            row[
                base
                + SQ_MUL2
                + B_OFF
                + i
            ]
            .into();

        builder.assert_zero(
            m2a.clone() - m2b
        );

        let m3a: AB::Expr =
            row[
                base
                + SQ_MUL3
                + A_OFF
                + i
            ]
            .into();

        builder.assert_zero(
            m3a - m1a
        );

        let m3b: AB::Expr =
            row[
                base
                + SQ_MUL3
                + B_OFF
                + i
            ]
            .into();

        builder.assert_zero(
            m3b - m2a
        );
    }

    // c0 + a1^2 = a0^2

    eval_modadd_at(
        builder,
        row,
        base + SQ_MODADD_START,
        base + SQ_C0,
        base + SQ_MUL2 + C_OFF,
        base + SQ_MUL1 + C_OFF,
    );

    // ab + ab = c1

    eval_modadd_at(
        builder,
        row,
        base
            + SQ_MODADD_START
            + MODADD_WIDTH,
        base + SQ_MUL3 + C_OFF,
        base + SQ_MUL3 + C_OFF,
        base + SQ_C1,
    );

    // Output range checks

    for idx in 0..SQ_EXTRA_LIMBS {
        let value_col =
            base + SQ_C0 + idx;

        let value: AB::Expr =
            row[value_col].into();

        let mut reconstructed =
            AB::Expr::ZERO;

        let mut power = 1u64;

        for bit in 0..LIMB_BITS {
            let col =
                base
                + SQ_OUTPUT_BITS
                + idx * LIMB_BITS
                + bit;

            let b: AB::Expr =
                row[col].into();

            builder.assert_zero(
                b.clone()
                    * (
                        b.clone()
                        - AB::Expr::ONE
                    )
            );

            reconstructed =
                reconstructed
                + AB::Expr::from_u64(
                    power
                ) * b;

            power <<= 1;
        }

        builder.assert_zero(
            value - reconstructed
        );
    }
}

// ============================================================
// Miller doubling-line AIR
// ============================================================

struct MillerDoubleAir;

impl<F> BaseAir<F>
    for MillerDoubleAir
{
    fn width(&self) -> usize {
        WIDTH
    }
}

impl<AB: AirBuilder> Air<AB>
    for MillerDoubleAir
{
    fn eval(
        &self,
        builder: &mut AB,
    ) {
        let main =
            builder.main();

        let row =
            main.current_slice();

        // ====================================================
        // B = Y^2
        // S2 = (Y+Z)^2
        // J = X^2
        // ====================================================

        eval_fq2_square(
            builder,
            row,
            SQ_B_BASE,
        );

        eval_fq2_square(
            builder,
            row,
            SQ_S2_BASE,
        );

        eval_fq2_square(
            builder,
            row,
            SQ_J_BASE,
        );

        // ====================================================
        // Fix first square input to standard G2 Y.
        // ====================================================

        for i in 0..LIMBS {
            let y0: AB::Expr =
                row[
                    SQ_B_BASE
                    + SQ_MUL1
                    + A_OFF
                    + i
                ]
                .into();

            builder.assert_zero(
                y0
                    - AB::Expr::from_u64(
                        G2_Y0[i]
                    )
            );

            let y1: AB::Expr =
                row[
                    SQ_B_BASE
                    + SQ_MUL2
                    + A_OFF
                    + i
                ]
                .into();

            builder.assert_zero(
                y1
                    - AB::Expr::from_u64(
                        G2_Y1[i]
                    )
            );
        }

        // ====================================================
        // Fix J square input to standard G2 X.
        // ====================================================

        for i in 0..LIMBS {
            let x0: AB::Expr =
                row[
                    SQ_J_BASE
                    + SQ_MUL1
                    + A_OFF
                    + i
                ]
                .into();

            builder.assert_zero(
                x0
                    - AB::Expr::from_u64(
                        G2_X0[i]
                    )
            );

            let x1: AB::Expr =
                row[
                    SQ_J_BASE
                    + SQ_MUL2
                    + A_OFF
                    + i
                ]
                .into();

            builder.assert_zero(
                x1
                    - AB::Expr::from_u64(
                        G2_X1[i]
                    )
            );
        }

        // ====================================================
        // S2 input must equal S = Y+Z.
        // ====================================================

        for i in 0..LIMBS {
            let s0: AB::Expr =
                row[
                    fq2_c0(S_START) + i
                ]
                .into();

            let sq_s0: AB::Expr =
                row[
                    SQ_S2_BASE
                    + SQ_MUL1
                    + A_OFF
                    + i
                ]
                .into();

            builder.assert_zero(
                sq_s0 - s0
            );

            let s1: AB::Expr =
                row[
                    fq2_c1(S_START) + i
                ]
                .into();

            let sq_s1: AB::Expr =
                row[
                    SQ_S2_BASE
                    + SQ_MUL2
                    + A_OFF
                    + i
                ]
                .into();

            builder.assert_zero(
                sq_s1 - s1
            );
        }

        // ====================================================
        // Fix Z=(1,0).
        // ====================================================

        for i in 0..LIMBS {
            let z0: AB::Expr =
                row[
                    fq2_c0(Z_START) + i
                ]
                .into();

            builder.assert_zero(
                z0
                    - AB::Expr::from_u64(
                        ONE_FQ[i]
                    )
            );

            let z1: AB::Expr =
                row[
                    fq2_c1(Z_START) + i
                ]
                .into();

            builder.assert_zero(z1);
        }

        // ====================================================
        // Fix E = 3*B2 for Z=1.
        // ====================================================

        for i in 0..LIMBS {
            let e0: AB::Expr =
                row[
                    fq2_c0(E_START) + i
                ]
                .into();

            builder.assert_zero(
                e0
                    - AB::Expr::from_u64(
                        E0[i]
                    )
            );

            let e1: AB::Expr =
                row[
                    fq2_c1(E_START) + i
                ]
                .into();

            builder.assert_zero(
                e1
                    - AB::Expr::from_u64(
                        E1[i]
                    )
            );
        }

        // ZERO=(0,0)

        for i in 0..FQ2_COLS {
            let z: AB::Expr =
                row[
                    ZERO_START + i
                ]
                .into();

            builder.assert_zero(z);
        }

        // ====================================================
        // Outer Fq2 relations
        // ====================================================

        let y0 =
            SQ_B_BASE
            + SQ_MUL1
            + A_OFF;

        let y1 =
            SQ_B_BASE
            + SQ_MUL2
            + A_OFF;

        let b0 =
            sq_out0(SQ_B_BASE);

        let b1 =
            sq_out1(SQ_B_BASE);

        let s20 =
            sq_out0(SQ_S2_BASE);

        let s21 =
            sq_out1(SQ_S2_BASE);

        let j0 =
            sq_out0(SQ_J_BASE);

        let j1 =
            sq_out1(SQ_J_BASE);

        // ----------------------------------------------------
        // Y + Z = S
        // ----------------------------------------------------

        eval_modadd_at(
            builder,
            row,
            OUTER_MODADD_START
                + 0 * MODADD_WIDTH,
            y0,
            fq2_c0(Z_START),
            fq2_c0(S_START),
        );

        eval_modadd_at(
            builder,
            row,
            OUTER_MODADD_START
                + 1 * MODADD_WIDTH,
            y1,
            fq2_c1(Z_START),
            fq2_c1(S_START),
        );

        // ----------------------------------------------------
        // B + Z = BC
        // ----------------------------------------------------

        eval_modadd_at(
            builder,
            row,
            OUTER_MODADD_START
                + 2 * MODADD_WIDTH,
            b0,
            fq2_c0(Z_START),
            fq2_c0(BC_START),
        );

        eval_modadd_at(
            builder,
            row,
            OUTER_MODADD_START
                + 3 * MODADD_WIDTH,
            b1,
            fq2_c1(Z_START),
            fq2_c1(BC_START),
        );

        // ----------------------------------------------------
        // H + BC = S2
        // ----------------------------------------------------

        eval_modadd_at(
            builder,
            row,
            OUTER_MODADD_START
                + 4 * MODADD_WIDTH,
            fq2_c0(H_START),
            fq2_c0(BC_START),
            s20,
        );

        eval_modadd_at(
            builder,
            row,
            OUTER_MODADD_START
                + 5 * MODADD_WIDTH,
            fq2_c1(H_START),
            fq2_c1(BC_START),
            s21,
        );

        // ----------------------------------------------------
        // I + B = E
        // ----------------------------------------------------

        eval_modadd_at(
            builder,
            row,
            OUTER_MODADD_START
                + 6 * MODADD_WIDTH,
            fq2_c0(I_START),
            b0,
            fq2_c0(E_START),
        );

        eval_modadd_at(
            builder,
            row,
            OUTER_MODADD_START
                + 7 * MODADD_WIDTH,
            fq2_c1(I_START),
            b1,
            fq2_c1(E_START),
        );

        // ----------------------------------------------------
        // J + J = 2J
        // ----------------------------------------------------

        eval_modadd_at(
            builder,
            row,
            OUTER_MODADD_START
                + 8 * MODADD_WIDTH,
            j0,
            j0,
            fq2_c0(TWOJ_START),
        );

        eval_modadd_at(
            builder,
            row,
            OUTER_MODADD_START
                + 9 * MODADD_WIDTH,
            j1,
            j1,
            fq2_c1(TWOJ_START),
        );

        // ----------------------------------------------------
        // 2J + J = 3J
        // ----------------------------------------------------

        eval_modadd_at(
            builder,
            row,
            OUTER_MODADD_START
                + 10 * MODADD_WIDTH,
            fq2_c0(TWOJ_START),
            j0,
            fq2_c0(THREEJ_START),
        );

        eval_modadd_at(
            builder,
            row,
            OUTER_MODADD_START
                + 11 * MODADD_WIDTH,
            fq2_c1(TWOJ_START),
            j1,
            fq2_c1(THREEJ_START),
        );

        // ----------------------------------------------------
        // -H + H = 0
        // ----------------------------------------------------

        eval_modadd_at(
            builder,
            row,
            OUTER_MODADD_START
                + 12 * MODADD_WIDTH,
            fq2_c0(NEG_H_START),
            fq2_c0(H_START),
            fq2_c0(ZERO_START),
        );

        eval_modadd_at(
            builder,
            row,
            OUTER_MODADD_START
                + 13 * MODADD_WIDTH,
            fq2_c1(NEG_H_START),
            fq2_c1(H_START),
            fq2_c1(ZERO_START),
        );

        // ====================================================
        // Range-check all extra Fq2 values.
        // ====================================================

        for idx in 0..EXTRA_LIMBS {
            let value_col =
                Z_START + idx;

            let value: AB::Expr =
                row[value_col].into();

            let mut reconstructed =
                AB::Expr::ZERO;

            let mut power = 1u64;

            for bit in 0..LIMB_BITS {
                let col =
                    EXTRA_BITS_START
                    + idx * LIMB_BITS
                    + bit;

                let b: AB::Expr =
                    row[col].into();

                builder.assert_zero(
                    b.clone()
                        * (
                            b.clone()
                            - AB::Expr::ONE
                        )
                );

                reconstructed =
                    reconstructed
                    + AB::Expr::from_u64(
                        power
                    ) * b;

                power <<= 1;
            }

            builder.assert_zero(
                value - reconstructed
            );
        }
    }
}

// ============================================================
// Big-integer witness helpers
// ============================================================

fn q_big() -> BigUint {
    limbs_to_big(&Q)
}

fn limbs_to_big(
    limbs: &Fq,
) -> BigUint {
    let mut value =
        BigUint::from(0u8);

    for i in (0..LIMBS).rev() {
        value <<= LIMB_BITS;
        value += limbs[i];
    }

    value
}

fn big_to_limbs(
    value: &BigUint,
) -> Fq {
    let mask =
        BigUint::from(0xffffu64);

    let mut x =
        value.clone();

    let mut out =
        [0u64; LIMBS];

    for i in 0..LIMBS {
        let digit =
            (&x & &mask)
                .to_u64_digits();

        out[i] =
            digit
                .first()
                .copied()
                .unwrap_or(0);

        x >>= LIMB_BITS;
    }

    assert_eq!(
        x,
        BigUint::from(0u8)
    );

    out
}

fn fq_add(
    a: &Fq,
    b: &Fq,
) -> Fq {
    let q = q_big();

    let mut s =
        limbs_to_big(a)
        + limbs_to_big(b);

    if s >= q {
        s -= &q;
    }

    big_to_limbs(&s)
}

fn fq_sub(
    a: &Fq,
    b: &Fq,
) -> Fq {
    let q = q_big();

    let aa =
        limbs_to_big(a);

    let bb =
        limbs_to_big(b);

    let value =
        if aa >= bb {
            aa - bb
        } else {
            aa + q - bb
        };

    big_to_limbs(&value)
}

fn fq_neg(
    a: &Fq,
) -> Fq {
    if *a == ZERO_FQ {
        ZERO_FQ
    } else {
        fq_sub(
            &ZERO_FQ,
            a,
        )
    }
}

fn fq_mul_witness(
    a: &Fq,
    b: &Fq,
) -> (
    Fq,
    Fq,
) {
    let q =
        q_big();

    let product =
        limbs_to_big(a)
        * limbs_to_big(b);

    let c =
        &product % &q;

    let k =
        &product / &q;

    (
        big_to_limbs(&c),
        big_to_limbs(&k),
    )
}

fn fq2_add(
    a: &Fq2,
    b: &Fq2,
) -> Fq2 {
    (
        fq_add(&a.0, &b.0),
        fq_add(&a.1, &b.1),
    )
}

fn fq2_sub(
    a: &Fq2,
    b: &Fq2,
) -> Fq2 {
    (
        fq_sub(&a.0, &b.0),
        fq_sub(&a.1, &b.1),
    )
}

fn fq2_neg(
    a: &Fq2,
) -> Fq2 {
    (
        fq_neg(&a.0),
        fq_neg(&a.1),
    )
}

fn fq2_square(
    a: &Fq2,
) -> Fq2 {
    let (a0_sq, _) =
        fq_mul_witness(
            &a.0,
            &a.0,
        );

    let (a1_sq, _) =
        fq_mul_witness(
            &a.1,
            &a.1,
        );

    let (a01, _) =
        fq_mul_witness(
            &a.0,
            &a.1,
        );

    (
        fq_sub(
            &a0_sq,
            &a1_sq,
        ),
        fq_add(
            &a01,
            &a01,
        ),
    )
}

// ============================================================
// Integer-schoolbook witness helpers
// ============================================================

fn multiply_limbs(
    a: &Fq,
    b: &Fq,
) -> (
    [u64; WIDE_LIMBS],
    [u64; 33],
) {
    let mut product =
        [0u64; WIDE_LIMBS];

    let mut carries =
        [0u64; 33];

    for out in 0..WIDE_LIMBS {
        let mut acc =
            carries[out] as u128;

        for i in 0..LIMBS {
            if out >= i {
                let j = out - i;

                if j < LIMBS {
                    acc +=
                        (a[i] as u128)
                        * (b[j] as u128);
                }
            }
        }

        product[out] =
            (
                acc
                % RADIX as u128
            ) as u64;

        carries[out + 1] =
            (
                acc
                / RADIX as u128
            ) as u64;
    }

    assert_eq!(
        carries[WIDE_LIMBS],
        0
    );

    (
        product,
        carries,
    )
}

fn add_c_plus_kq(
    c: &Fq,
    kq: &[u64; WIDE_LIMBS],
    expected: &[u64; WIDE_LIMBS],
) -> [u64; 33] {
    let mut carries =
        [0u64; 33];

    for i in 0..WIDE_LIMBS {
        let c_i =
            if i < LIMBS {
                c[i]
            } else {
                0
            };

        let acc =
            c_i as u128
            + kq[i] as u128
            + carries[i] as u128;

        let limb =
            (
                acc
                % RADIX as u128
            ) as u64;

        carries[i + 1] =
            (
                acc
                / RADIX as u128
            ) as u64;

        assert_eq!(
            limb,
            expected[i]
        );
    }

    assert_eq!(
        carries[WIDE_LIMBS],
        0
    );

    carries
}

fn push_bits(
    row: &mut Vec<Val>,
    value: u64,
    bits: usize,
) {
    assert!(
        value < (1u64 << bits)
    );

    for i in 0..bits {
        row.push(
            Val::from_u64(
                (value >> i) & 1
            )
        );
    }
}

// ============================================================
// Build one Fq multiplication witness
// ============================================================

fn build_modmul_block(
    a: Fq,
    b: Fq,
) -> (
    Vec<Val>,
    Fq,
) {
    let (
        c,
        k,
    ) =
        fq_mul_witness(
            &a,
            &b,
        );

    let (
        ab,
        ab_carries,
    ) =
        multiply_limbs(
            &a,
            &b,
        );

    let (
        kq,
        kq_carries,
    ) =
        multiply_limbs(
            &k,
            &Q,
        );

    let add_carries =
        add_c_plus_kq(
            &c,
            &kq,
            &ab,
        );

    let mut row =
        Vec::with_capacity(
            MODMUL_WIDTH
        );

    for x in a {
        row.push(Val::from_u64(x));
    }

    for x in b {
        row.push(Val::from_u64(x));
    }

    for x in c {
        row.push(Val::from_u64(x));
    }

    for x in k {
        row.push(Val::from_u64(x));
    }

    for x in ab {
        row.push(Val::from_u64(x));
    }

    for x in ab_carries {
        row.push(Val::from_u64(x));
    }

    for x in kq {
        row.push(Val::from_u64(x));
    }

    for x in kq_carries {
        row.push(Val::from_u64(x));
    }

    for x in add_carries {
        row.push(Val::from_u64(x));
    }

    for x in a {
        push_bits(
            &mut row,
            x,
            LIMB_BITS,
        );
    }

    for x in b {
        push_bits(
            &mut row,
            x,
            LIMB_BITS,
        );
    }

    for x in c {
        push_bits(
            &mut row,
            x,
            LIMB_BITS,
        );
    }

    for x in k {
        push_bits(
            &mut row,
            x,
            LIMB_BITS,
        );
    }

    for x in ab {
        push_bits(
            &mut row,
            x,
            LIMB_BITS,
        );
    }

    for x in kq {
        push_bits(
            &mut row,
            x,
            LIMB_BITS,
        );
    }

    for x in ab_carries {
        push_bits(
            &mut row,
            x,
            CARRY_BITS,
        );
    }

    for x in kq_carries {
        push_bits(
            &mut row,
            x,
            CARRY_BITS,
        );
    }

    assert_eq!(
        row.len(),
        MODMUL_WIDTH
    );

    (
        row,
        c,
    )
}

// ============================================================
// Fq modular-add witness
// ============================================================

fn addition_carries(
    a: &Fq,
    b: &Fq,
) -> [u64; MODADD_CARRY_COUNT] {
    let mut carry =
        [0u64; MODADD_CARRY_COUNT];

    for i in 0..LIMBS {
        let v =
            a[i] as u128
            + b[i] as u128
            + carry[i] as u128;

        carry[i + 1] =
            (
                v
                / RADIX as u128
            ) as u64;
    }

    carry
}

fn rhs_carries(
    c: &Fq,
    reduction: u64,
) -> [u64; MODADD_CARRY_COUNT] {
    let mut carry =
        [0u64; MODADD_CARRY_COUNT];

    for i in 0..LIMBS {
        let v =
            c[i] as u128
            + reduction as u128
                * Q[i] as u128
            + carry[i] as u128;

        carry[i + 1] =
            (
                v
                / RADIX as u128
            ) as u64;
    }

    carry
}

fn push_modadd(
    row: &mut Vec<Val>,
    a: &Fq,
    b: &Fq,
    c: &Fq,
) {
    let q =
        q_big();

    let sum =
        limbs_to_big(a)
        + limbs_to_big(b);

    let expected =
        &sum % &q;

    assert_eq!(
        big_to_limbs(&expected),
        *c
    );

    let reduction =
        if sum >= q {
            1u64
        } else {
            0u64
        };

    let lhs =
        addition_carries(
            a,
            b,
        );

    let rhs =
        rhs_carries(
            c,
            reduction,
        );

    for i in 0..LIMBS {
        let lhs_digit =
            a[i] as i128
            + b[i] as i128
            + lhs[i] as i128
            - RADIX as i128
                * lhs[i + 1]
                    as i128;

        let rhs_digit =
            c[i] as i128
            + reduction as i128
                * Q[i] as i128
            + rhs[i] as i128
            - RADIX as i128
                * rhs[i + 1]
                    as i128;

        assert_eq!(
            lhs_digit,
            rhs_digit
        );
    }

    row.push(
        Val::from_u64(
            reduction
        )
    );

    for x in lhs {
        row.push(
            Val::from_u64(x)
        );
    }

    for x in rhs {
        row.push(
            Val::from_u64(x)
        );
    }
}

// ============================================================
// Build an Fq2-square block
// ============================================================

fn build_fq2_square_block(
    input: Fq2,
) -> (
    Vec<Val>,
    Fq2,
) {
    let (
        m1,
        a0_sq,
    ) =
        build_modmul_block(
            input.0,
            input.0,
        );

    let (
        m2,
        a1_sq,
    ) =
        build_modmul_block(
            input.1,
            input.1,
        );

    let (
        m3,
        a01,
    ) =
        build_modmul_block(
            input.0,
            input.1,
        );

    let c0 =
        fq_sub(
            &a0_sq,
            &a1_sq,
        );

    let c1 =
        fq_add(
            &a01,
            &a01,
        );

    let mut row =
        Vec::with_capacity(
            SQ_WIDTH
        );

    row.extend_from_slice(&m1);
    row.extend_from_slice(&m2);
    row.extend_from_slice(&m3);

    for x in c0 {
        row.push(
            Val::from_u64(x)
        );
    }

    for x in c1 {
        row.push(
            Val::from_u64(x)
        );
    }

    // c0 + a1^2 = a0^2
    push_modadd(
        &mut row,
        &c0,
        &a1_sq,
        &a0_sq,
    );

    // a01+a01 = c1
    push_modadd(
        &mut row,
        &a01,
        &a01,
        &c1,
    );

    for x in c0 {
        push_bits(
            &mut row,
            x,
            LIMB_BITS,
        );
    }

    for x in c1 {
        push_bits(
            &mut row,
            x,
            LIMB_BITS,
        );
    }

    assert_eq!(
        row.len(),
        SQ_WIDTH
    );

    (
        row,
        (c0, c1),
    )
}

fn push_fq2(
    row: &mut Vec<Val>,
    x: &Fq2,
) {
    for v in x.0 {
        row.push(
            Val::from_u64(v)
        );
    }

    for v in x.1 {
        row.push(
            Val::from_u64(v)
        );
    }
}

fn push_fq2_bits(
    row: &mut Vec<Val>,
    x: &Fq2,
) {
    for v in x.0 {
        push_bits(
            row,
            v,
            LIMB_BITS,
        );
    }

    for v in x.1 {
        push_bits(
            row,
            v,
            LIMB_BITS,
        );
    }
}

// ============================================================
// Trace construction
// ============================================================

fn make_trace(
    height: usize,
) -> RowMajorMatrix<Val> {
    let x: Fq2 =
        (
            G2_X0,
            G2_X1,
        );

    let y: Fq2 =
        (
            G2_Y0,
            G2_Y1,
        );

    let z: Fq2 =
        (
            ONE_FQ,
            ZERO_FQ,
        );

    let zero: Fq2 =
        (
            ZERO_FQ,
            ZERO_FQ,
        );

    let e: Fq2 =
        (
            E0,
            E1,
        );

    // B = Y^2
    let (
        block_b,
        b,
    ) =
        build_fq2_square_block(
            y
        );

    // S = Y+Z
    let s =
        fq2_add(
            &y,
            &z,
        );

    // S2=(Y+Z)^2
    let (
        block_s2,
        s2,
    ) =
        build_fq2_square_block(
            s
        );

    // J=X^2
    let (
        block_j,
        j,
    ) =
        build_fq2_square_block(
            x
        );

    // B+C where C=1
    let bc =
        fq2_add(
            &b,
            &z,
        );

    // H=S2-(B+C)
    let h =
        fq2_sub(
            &s2,
            &bc,
        );

    // I=E-B
    let i =
        fq2_sub(
            &e,
            &b,
        );

    let twoj =
        fq2_add(
            &j,
            &j,
        );

    let threej =
        fq2_add(
            &twoj,
            &j,
        );

    let neg_h =
        fq2_neg(
            &h,
        );

    let mut one_row =
        Vec::with_capacity(
            WIDTH
        );

    one_row.extend_from_slice(
        &block_b
    );

    one_row.extend_from_slice(
        &block_s2
    );

    one_row.extend_from_slice(
        &block_j
    );

    // Extra Fq2 values, exact layout order.

    push_fq2(
        &mut one_row,
        &z,
    );

    push_fq2(
        &mut one_row,
        &s,
    );

    push_fq2(
        &mut one_row,
        &bc,
    );

    push_fq2(
        &mut one_row,
        &h,
    );

    push_fq2(
        &mut one_row,
        &i,
    );

    push_fq2(
        &mut one_row,
        &twoj,
    );

    push_fq2(
        &mut one_row,
        &threej,
    );

    push_fq2(
        &mut one_row,
        &neg_h,
    );

    push_fq2(
        &mut one_row,
        &e,
    );

    push_fq2(
        &mut one_row,
        &zero,
    );

    // ========================================================
    // 14 outer Fq modular additions
    // ========================================================

    // Y+Z=S
    push_modadd(
        &mut one_row,
        &y.0,
        &z.0,
        &s.0,
    );

    push_modadd(
        &mut one_row,
        &y.1,
        &z.1,
        &s.1,
    );

    // B+Z=BC
    push_modadd(
        &mut one_row,
        &b.0,
        &z.0,
        &bc.0,
    );

    push_modadd(
        &mut one_row,
        &b.1,
        &z.1,
        &bc.1,
    );

    // H+BC=S2
    push_modadd(
        &mut one_row,
        &h.0,
        &bc.0,
        &s2.0,
    );

    push_modadd(
        &mut one_row,
        &h.1,
        &bc.1,
        &s2.1,
    );

    // I+B=E
    push_modadd(
        &mut one_row,
        &i.0,
        &b.0,
        &e.0,
    );

    push_modadd(
        &mut one_row,
        &i.1,
        &b.1,
        &e.1,
    );

    // J+J=2J
    push_modadd(
        &mut one_row,
        &j.0,
        &j.0,
        &twoj.0,
    );

    push_modadd(
        &mut one_row,
        &j.1,
        &j.1,
        &twoj.1,
    );

    // 2J+J=3J
    push_modadd(
        &mut one_row,
        &twoj.0,
        &j.0,
        &threej.0,
    );

    push_modadd(
        &mut one_row,
        &twoj.1,
        &j.1,
        &threej.1,
    );

    // -H+H=0
    push_modadd(
        &mut one_row,
        &neg_h.0,
        &h.0,
        &zero.0,
    );

    push_modadd(
        &mut one_row,
        &neg_h.1,
        &h.1,
        &zero.1,
    );

    // ========================================================
    // Extra-value range bits
    // ========================================================

    for value in [
        z,
        s,
        bc,
        h,
        i,
        twoj,
        threej,
        neg_h,
        e,
        zero,
    ] {
        push_fq2_bits(
            &mut one_row,
            &value,
        );
    }

    assert_eq!(
        one_row.len(),
        WIDTH,
        "trace width mismatch: {} != {}",
        one_row.len(),
        WIDTH
    );

    // Print line coefficients once from witness.

    println!(
        "Computed Miller line coefficients:"
    );

    println!(
        "  ell_0 = -H"
    );

    println!(
        "  ell_1 = 3J"
    );

    println!(
        "  ell_2 = I"
    );

    println!();

    let mut values =
        Vec::with_capacity(
            WIDTH * height
        );

    for _ in 0..height {
        values.extend_from_slice(
            &one_row
        );
    }

    RowMajorMatrix::new(
        values,
        WIDTH,
    )
}

// ============================================================
// Main
// ============================================================

fn main() {
    println!(
        "=============================================="
    );

    println!(
        " Plonky3 BN254 Miller Doubling Line"
    );

    println!(
        "=============================================="
    );

    println!();

    println!(
        "Native proving field : Goldilocks"
    );

    println!(
        "Pairing curve        : BN254"
    );

    println!(
        "G2 coordinates       : Fq2"
    );

    println!(
        "Twist type           : D"
    );

    println!();

    println!(
        "This checkpoint proves:"
    );

    println!(
        "  B = Y^2"
    );

    println!(
        "  H = (Y+Z)^2 - (B+C)"
    );

    println!(
        "  I = E-B"
    );

    println!(
        "  J = X^2"
    );

    println!();

    println!(
        "and outputs:"
    );

    println!(
        "  ell_0 = -H"
    );

    println!(
        "  ell_1 = 3J"
    );

    println!(
        "  ell_2 = I"
    );

    println!();

    println!(
        "Initial G2 state uses the standard generator and Z=1."
    );

    println!();

    let trace_height =
        8usize;

    let trace =
        make_trace(
            trace_height
        );

    // ========================================================
    // Plonky3 configuration
    // ========================================================

    type ByteHash =
        Sha256;

    type FieldHash =
        SerializingHasher<
            ByteHash
        >;

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

    type Config =
        StarkConfig<
            Pcs,
            Challenge,
            Challenger,
        >;

    let config =
        Config::new(
            pcs,
            challenger,
        );

    // ========================================================
    // Prove
    // ========================================================

    let prove_start =
        Instant::now();

    let proof =
        prove(
            &config,
            &MillerDoubleAir,
            trace,
            &[],
        )
        .expect(
            "proof generation failed"
        );

    let prove_time =
        prove_start.elapsed();

    let proof_bytes =
        postcard::to_allocvec(
            &proof
        )
        .expect(
            "proof serialization failed"
        );

    // ========================================================
    // Verify
    // ========================================================

    let verify_start =
        Instant::now();

    verify(
        &config,
        &MillerDoubleAir,
        &proof,
        &[],
    )
    .expect(
        "verification failed"
    );

    let verify_time =
        verify_start.elapsed();

    // ========================================================
    // Metrics
    // ========================================================

    const FQ2_SQUARE_CONSTRAINTS: usize =
        11800;

    const OUTER_MODADD_CONSTRAINTS: usize =
        OUTER_MODADDS * 52;

    // Fixed:
    // Y = 32
    // X = 32
    // S-square link = 32
    // Z = 32
    // E = 32
    // ZERO = 32
    const LINK_AND_CONSTANT_CONSTRAINTS: usize =
        192;

    const EXTRA_RANGE_CONSTRAINTS: usize =
        EXTRA_LIMBS
        * (LIMB_BITS + 1);

    let total_constraints =
        3 * FQ2_SQUARE_CONSTRAINTS
        + OUTER_MODADD_CONSTRAINTS
        + LINK_AND_CONSTANT_CONSTRAINTS
        + EXTRA_RANGE_CONSTRAINTS;

    println!(
        "=============================================="
    );

    println!(
        " Metrics"
    );

    println!(
        "=============================================="
    );

    println!();

    println!(
        "Fq2 squarings             : 3"
    );

    println!(
        "Outer Fq modular adds     : {}",
        OUTER_MODADDS
    );

    println!(
        "Constraints / Fq2 square : {}",
        FQ2_SQUARE_CONSTRAINTS
    );

    println!(
        "Outer modadd constraints  : {}",
        OUTER_MODADD_CONSTRAINTS
    );

    println!(
        "Link/constant constraints : {}",
        LINK_AND_CONSTANT_CONSTRAINTS
    );

    println!(
        "Extra range constraints   : {}",
        EXTRA_RANGE_CONSTRAINTS
    );

    println!(
        "AIR constraint expressions : {}",
        total_constraints
    );

    println!(
        "Maximum constraint degree  : 2"
    );

    println!(
        "Trace width                : {} columns",
        WIDTH
    );

    println!(
        "Trace height               : {} rows",
        trace_height
    );

    println!(
        "Trace cells                : {}",
        WIDTH * trace_height
    );

    println!(
        "Row-wise evaluations       : {}",
        total_constraints
            * trace_height
    );

    println!(
        "Proof size                 : {} bytes",
        proof_bytes.len()
    );

    println!(
        "Proving time               : {:?}",
        prove_time
    );

    println!(
        "Verification time          : {:?}",
        verify_time
    );

    println!();

    println!(
        "BN254 Miller doubling-line proof verified!"
    );
}