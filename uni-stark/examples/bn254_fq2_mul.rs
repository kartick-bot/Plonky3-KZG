use std::time::Instant;

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

// ============================================================
// BN254 base field modulus q
// ============================================================

const Q: [u64; LIMBS] = [
    64839, 55420, 35862, 15392,
    51853, 26737, 27281, 38785,
    22621, 33153, 17846, 47184,
    41001, 57649, 20082, 12388,
];

const ZERO: [u64; LIMBS] = [0; LIMBS];

const ONE: [u64; LIMBS] = [
    1, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
];

const TWO: [u64; LIMBS] = [
    2, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
];

const THREE: [u64; LIMBS] = [
    3, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
];

const FOUR: [u64; LIMBS] = [
    4, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
];

const SIX: [u64; LIMBS] = [
    6, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
];

const EIGHT: [u64; LIMBS] = [
    8, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
];

const TEN: [u64; LIMBS] = [
    10, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
];

// q - 5
const Q_MINUS_5: [u64; LIMBS] = [
    64834, 55420, 35862, 15392,
    51853, 26737, 27281, 38785,
    22621, 33153, 17846, 47184,
    41001, 57649, 20082, 12388,
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

const AB_OFF: usize =
    K_OFF + LIMBS;

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

const BLOCK_WIDTH: usize =
    CARRY_BIT_OFF + CARRY_BIT_COLS;

// ============================================================
// Four Fq multiplications
//
// M1 = a0*b0
// M2 = a1*b1
// M3 = a0*b1
// M4 = a1*b0
// ============================================================

const MUL1_BASE: usize = 0;

const MUL2_BASE: usize =
    MUL1_BASE + BLOCK_WIDTH;

const MUL3_BASE: usize =
    MUL2_BASE + BLOCK_WIDTH;

const MUL4_BASE: usize =
    MUL3_BASE + BLOCK_WIDTH;

const MUL_END: usize =
    MUL4_BASE + BLOCK_WIDTH;

// ============================================================
// Fq2 outputs
//
// c0 + c1*u
// ============================================================

const C0_START: usize =
    MUL_END;

const C1_START: usize =
    C0_START + LIMBS;

// ============================================================
// Two modular-add relations
//
// R0:
//     c0 + t11 = t00 (mod q)
//
// R1:
//     t01 + t10 = c1 (mod q)
// ============================================================

const NUM_MODADDS: usize = 2;

const MODADD_CARRY_COUNT: usize = 17;

// reduction bit
// lhs carry[17]
// rhs carry[17]
const MODADD_WIDTH: usize =
    1
    + MODADD_CARRY_COUNT
    + MODADD_CARRY_COUNT;

const MODADD_START: usize =
    C1_START + LIMBS;

const MODADD_END: usize =
    MODADD_START
    + NUM_MODADDS * MODADD_WIDTH;

// ============================================================
// Range checks for c0,c1
// ============================================================

const EXTRA_LIMBS: usize =
    2 * LIMBS;

const EXTRA_BIT_START: usize =
    MODADD_END;

const EXTRA_BIT_COLS: usize =
    EXTRA_LIMBS * LIMB_BITS;

const WIDTH: usize =
    EXTRA_BIT_START
    + EXTRA_BIT_COLS;

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

    let a_start =
        base + A_OFF;

    let b_start =
        base + B_OFF;

    let c_start =
        base + C_OFF;

    let k_start =
        base + K_OFF;

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

    // ========================================================
    // a*b
    // ========================================================

    let carry0: AB::Expr =
        row[ab_carry_start].into();

    builder.assert_zero(
        carry0
    );

    for out in 0..WIDE_LIMBS {
        let mut acc: AB::Expr =
            row[
                ab_carry_start + out
            ]
            .into();

        for i in 0..LIMBS {
            if out >= i {
                let j =
                    out - i;

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
                        acc
                        + a_i * b_j;
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

    builder.assert_zero(
        final_ab
    );

    // ========================================================
    // k*q
    // ========================================================

    let kq0: AB::Expr =
        row[kq_carry_start].into();

    builder.assert_zero(
        kq0
    );

    for out in 0..WIDE_LIMBS {
        let mut acc: AB::Expr =
            row[
                kq_carry_start
                + out
            ]
            .into();

        for i in 0..LIMBS {
            if out >= i {
                let j =
                    out - i;

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
                        acc
                        + k_i * q_j;
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

    builder.assert_zero(
        final_kq
    );

    // ========================================================
    // c + kq = ab
    // ========================================================

    let add0: AB::Expr =
        row[
            add_carry_start
        ]
        .into();

    builder.assert_zero(
        add0
    );

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

    builder.assert_zero(
        final_add
    );

    // ========================================================
    // 16-bit limb range checks
    // ========================================================

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
                < 4 * LIMBS
                + WIDE_LIMBS
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
            row[
                value_col
            ]
            .into();

        let mut reconstructed =
            AB::Expr::ZERO;

        let mut power =
            1u64;

        for bit in 0..LIMB_BITS {
            let col =
                limb_bit_start
                + range_idx
                    * LIMB_BITS
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
            value
                - reconstructed
        );
    }

    // ========================================================
    // Multiplication carry range checks
    // ========================================================

    for idx in 0..MUL_CARRIES {
        let value_col =
            if idx < 33 {
                ab_carry_start
                    + idx
            } else {
                kq_carry_start
                    + idx
                    - 33
            };

        let value: AB::Expr =
            row[
                value_col
            ]
            .into();

        let mut reconstructed =
            AB::Expr::ZERO;

        let mut power =
            1u64;

        for bit in 0..CARRY_BITS {
            let col =
                carry_bit_start
                + idx
                    * CARRY_BITS
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
            value
                - reconstructed
        );
    }
}

// ============================================================
// Modular addition
//
// a+b = c+r*q
// ============================================================

fn eval_modadd<AB: AirBuilder>(
    builder: &mut AB,
    row: &[AB::Var],
    relation: usize,
    a_start: usize,
    b_start: usize,
    c_start: usize,
) {
    let base =
        MODADD_START
        + relation * MODADD_WIDTH;

    let r_col =
        base;

    let lhs_carry_start =
        base + 1;

    let rhs_carry_start =
        lhs_carry_start
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
        row[
            lhs_carry_start
        ]
        .into();

    let rhs0: AB::Expr =
        row[
            rhs_carry_start
        ]
        .into();

    builder.assert_zero(lhs0);
    builder.assert_zero(rhs0);

    let radix =
        AB::Expr::from_u64(
            RADIX
        );

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
                lhs_carry_start + i
            ]
            .into();

        let lhs_out: AB::Expr =
            row[
                lhs_carry_start
                + i
                + 1
            ]
            .into();

        let rhs_in: AB::Expr =
            row[
                rhs_carry_start + i
            ]
            .into();

        let rhs_out: AB::Expr =
            row[
                rhs_carry_start
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
            lhs_carry_start
            + LIMBS
        ]
        .into();

    let rhs_final: AB::Expr =
        row[
            rhs_carry_start
            + LIMBS
        ]
        .into();

    builder.assert_zero(
        lhs_final
            - rhs_final
    );
}

// ============================================================
// Fq2 multiplication AIR
// ============================================================

struct Fq2MulAir;

impl<F> BaseAir<F>
    for Fq2MulAir
{
    fn width(&self) -> usize {
        WIDTH
    }
}

impl<AB: AirBuilder> Air<AB>
    for Fq2MulAir
{
    fn eval(
        &self,
        builder: &mut AB,
    ) {
        let main =
            builder.main();

        let row =
            main.current_slice();

        // ----------------------------------------------------
        // Four Fq multiplications
        // ----------------------------------------------------

        eval_modmul(
            builder,
            row,
            MUL1_BASE,
        );

        eval_modmul(
            builder,
            row,
            MUL2_BASE,
        );

        eval_modmul(
            builder,
            row,
            MUL3_BASE,
        );

        eval_modmul(
            builder,
            row,
            MUL4_BASE,
        );

        // ----------------------------------------------------
        // Dynamic input linking
        //
        // M1: a0*b0
        // M2: a1*b1
        // M3: a0*b1
        // M4: a1*b0
        // ----------------------------------------------------

        for i in 0..LIMBS {
            // M3.a = M1.a = a0

            let m3a: AB::Expr =
                row[
                    MUL3_BASE
                    + A_OFF
                    + i
                ]
                .into();

            let m1a: AB::Expr =
                row[
                    MUL1_BASE
                    + A_OFF
                    + i
                ]
                .into();

            builder.assert_zero(
                m3a - m1a
            );

            // M3.b = M2.b = b1

            let m3b: AB::Expr =
                row[
                    MUL3_BASE
                    + B_OFF
                    + i
                ]
                .into();

            let m2b: AB::Expr =
                row[
                    MUL2_BASE
                    + B_OFF
                    + i
                ]
                .into();

            builder.assert_zero(
                m3b - m2b
            );

            // M4.a = M2.a = a1

            let m4a: AB::Expr =
                row[
                    MUL4_BASE
                    + A_OFF
                    + i
                ]
                .into();

            let m2a: AB::Expr =
                row[
                    MUL2_BASE
                    + A_OFF
                    + i
                ]
                .into();

            builder.assert_zero(
                m4a - m2a
            );

            // M4.b = M1.b = b0

            let m4b: AB::Expr =
                row[
                    MUL4_BASE
                    + B_OFF
                    + i
                ]
                .into();

            let m1b: AB::Expr =
                row[
                    MUL1_BASE
                    + B_OFF
                    + i
                ]
                .into();

            builder.assert_zero(
                m4b - m1b
            );
        }

        // ----------------------------------------------------
        // Real component:
        //
        // c0 = t00 - t11
        //
        // equivalently:
        //
        // c0 + t11 = t00 mod q
        // ----------------------------------------------------

        eval_modadd(
            builder,
            row,
            0,
            C0_START,
            MUL2_BASE + C_OFF,
            MUL1_BASE + C_OFF,
        );

        // ----------------------------------------------------
        // Imaginary component:
        //
        // c1 = t01 + t10 mod q
        // ----------------------------------------------------

        eval_modadd(
            builder,
            row,
            1,
            MUL3_BASE + C_OFF,
            MUL4_BASE + C_OFF,
            C1_START,
        );

        // ----------------------------------------------------
        // Range-check c0 and c1
        // ----------------------------------------------------

        for idx in 0..EXTRA_LIMBS {
            let value_col =
                C0_START + idx;

            let value: AB::Expr =
                row[
                    value_col
                ]
                .into();

            let mut reconstructed =
                AB::Expr::ZERO;

            let mut power =
                1u64;

            for bit in 0..LIMB_BITS {
                let col =
                    EXTRA_BIT_START
                    + idx
                        * LIMB_BITS
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
                value
                    - reconstructed
            );
        }
    }
}

// ============================================================
// Witness helpers
// ============================================================

fn multiply_limbs(
    a: &[u64; LIMBS],
    b: &[u64; LIMBS],
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
                let j =
                    out - i;

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
    c: &[u64; LIMBS],
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

fn build_modmul_block(
    a: [u64; LIMBS],
    b: [u64; LIMBS],
    c: [u64; LIMBS],
    k: [u64; LIMBS],
) -> Vec<Val> {
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
            BLOCK_WIDTH
        );

    for x in a {
        row.push(
            Val::from_u64(x)
        );
    }

    for x in b {
        row.push(
            Val::from_u64(x)
        );
    }

    for x in c {
        row.push(
            Val::from_u64(x)
        );
    }

    for x in k {
        row.push(
            Val::from_u64(x)
        );
    }

    for x in ab {
        row.push(
            Val::from_u64(x)
        );
    }

    for x in ab_carries {
        row.push(
            Val::from_u64(x)
        );
    }

    for x in kq {
        row.push(
            Val::from_u64(x)
        );
    }

    for x in kq_carries {
        row.push(
            Val::from_u64(x)
        );
    }

    for x in add_carries {
        row.push(
            Val::from_u64(x)
        );
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
        BLOCK_WIDTH
    );

    row
}

// ============================================================
// Modular addition witness
// ============================================================

fn addition_carries(
    a: &[u64; LIMBS],
    b: &[u64; LIMBS],
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

fn addition_rhs_carries(
    c: &[u64; LIMBS],
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

fn push_modadd_witness(
    row: &mut Vec<Val>,
    a: &[u64; LIMBS],
    b: &[u64; LIMBS],
    c: &[u64; LIMBS],
    reduction: u64,
) {
    let lhs =
        addition_carries(
            a,
            b,
        );

    let rhs =
        addition_rhs_carries(
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

    assert_eq!(
        lhs[LIMBS],
        rhs[LIMBS]
    );

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
// Trace
// ============================================================

fn make_trace(
    height: usize,
) -> RowMajorMatrix<Val> {
    // A = 1 + 2u
    //
    // B = 3 + 4u

    let a0 =
        ONE;

    let a1 =
        TWO;

    let b0 =
        THREE;

    let b1 =
        FOUR;

    // t00 = 1*3 = 3
    let t00 =
        THREE;

    // t11 = 2*4 = 8
    let t11 =
        EIGHT;

    // t01 = 1*4 = 4
    let t01 =
        FOUR;

    // t10 = 2*3 = 6
    let t10 =
        SIX;

    // c0 = 3-8 mod q = q-5
    let c0 =
        Q_MINUS_5;

    // c1 = 4+6 = 10
    let c1 =
        TEN;

    let block1 =
        build_modmul_block(
            a0,
            b0,
            t00,
            ZERO,
        );

    let block2 =
        build_modmul_block(
            a1,
            b1,
            t11,
            ZERO,
        );

    let block3 =
        build_modmul_block(
            a0,
            b1,
            t01,
            ZERO,
        );

    let block4 =
        build_modmul_block(
            a1,
            b0,
            t10,
            ZERO,
        );

    let mut one_row =
        Vec::with_capacity(
            WIDTH
        );

    one_row.extend_from_slice(
        &block1
    );

    one_row.extend_from_slice(
        &block2
    );

    one_row.extend_from_slice(
        &block3
    );

    one_row.extend_from_slice(
        &block4
    );

    // c0
    for x in c0 {
        one_row.push(
            Val::from_u64(x)
        );
    }

    // c1
    for x in c1 {
        one_row.push(
            Val::from_u64(x)
        );
    }

    // R0:
    //
    // c0 + t11
    // =
    // t00 + q
    //
    // reduction = 1

    push_modadd_witness(
        &mut one_row,
        &c0,
        &t11,
        &t00,
        1,
    );

    // R1:
    //
    // t01+t10 = c1

    push_modadd_witness(
        &mut one_row,
        &t01,
        &t10,
        &c1,
        0,
    );

    // Range-check c0/c1

    for x in c0 {
        push_bits(
            &mut one_row,
            x,
            LIMB_BITS,
        );
    }

    for x in c1 {
        push_bits(
            &mut one_row,
            x,
            LIMB_BITS,
        );
    }

    assert_eq!(
        one_row.len(),
        WIDTH,
        "trace width mismatch: {} != {}",
        one_row.len(),
        WIDTH
    );

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
        " Plonky3 BN254 Fq2 Multiplication"
    );

    println!(
        "=============================================="
    );

    println!();

    println!(
        "Native proving field : Goldilocks"
    );

    println!(
        "Target field         : BN254 Fq2"
    );

    println!(
        "Non-residue          : u^2 = -1"
    );

    println!();

    println!(
        "Test:"
    );

    println!(
        "  A = 1 + 2u"
    );

    println!(
        "  B = 3 + 4u"
    );

    println!();

    println!(
        "Expected:"
    );

    println!(
        "  real = 3 - 8 = -5 mod q"
    );

    println!(
        "  imag = 4 + 6 = 10"
    );

    println!();

    let trace_height =
        8usize;

    let trace =
        make_trace(
            trace_height
        );

    // ========================================================
    // Merkle / FRI
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
            &Fq2MulAir,
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
        &Fq2MulAir,
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

    const MODMUL_CONSTRAINTS: usize =
        3696;

    // Four 16-limb cross-block equalities.
    const LINK_CONSTRAINTS: usize =
        4 * LIMBS;

    // Each modular addition:
    //
    // reduction boolean          1
    // starting carries           2
    // limb equations            16
    // carry booleans            32
    // final carry equality       1
    //
    // total = 52

    const MODADD_CONSTRAINTS_PER: usize =
        52;

    const MODADD_CONSTRAINTS: usize =
        NUM_MODADDS
        * MODADD_CONSTRAINTS_PER;

    // c0,c1:
    // 32 limbs
    // each has:
    // 16 boolean bits + reconstruction
    const EXTRA_RANGE_CONSTRAINTS: usize =
        EXTRA_LIMBS
        * (LIMB_BITS + 1);

    let total_constraints =
        4 * MODMUL_CONSTRAINTS
        + LINK_CONSTRAINTS
        + MODADD_CONSTRAINTS
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
        "Fq multiplications         : 4"
    );

    println!(
        "Fq modular additions       : 2"
    );

    println!(
        "Constraints / Fq mul       : {}",
        MODMUL_CONSTRAINTS
    );

    println!(
        "Input link constraints     : {}",
        LINK_CONSTRAINTS
    );

    println!(
        "Modadd constraints         : {}",
        MODADD_CONSTRAINTS
    );

    println!(
        "Output range constraints   : {}",
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
        "BN254 Fq2 multiplication proof verified!"
    );
}