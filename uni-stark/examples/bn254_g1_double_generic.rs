use std::time::Instant;

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
// BN254 non-native representation
// ============================================================

const LIMBS: usize = 16;
const WIDE_LIMBS: usize = 32;

const LIMB_BITS: usize = 16;
const CARRY_BITS: usize = 20;

const RADIX: u64 = 1 << LIMB_BITS;

// ============================================================
// BN254 base-field modulus q
// ============================================================

const Q: [u64; LIMBS] = [
    64839, 55420, 35862, 15392, 51853, 26737, 27281, 38785, 22621, 33153, 17846, 47184, 41001,
    57649, 20082, 12388,
];

// ============================================================
// Test witness values.
//
// IMPORTANT:
//
// These are used ONLY by make_trace().
//
// The AIR below does not force the point to equal these values.
// ============================================================

const ZERO: [u64; LIMBS] = [0; LIMBS];

const ONE: [u64; LIMBS] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

const TWO: [u64; LIMBS] = [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

const THREE: [u64; LIMBS] = [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

const FOUR: [u64; LIMBS] = [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

// lambda = 3/4 mod q
const TEST_LAMBDA: [u64; LIMBS] = [
    48630, 8797, 26897, 60696, 22505, 3669, 4077, 12705, 582, 57633, 13384, 19004, 14367, 10469,
    15062, 9291,
];

// x-coordinate of 2*(1,2)
const TEST_X3: [u64; LIMBS] = [
    53203, 28039, 2241, 54210, 7336, 5767, 5801, 55672, 5509, 26648, 1115, 39813, 6658, 11795,
    17639, 774,
];

// y-coordinate of 2*(1,2)
const TEST_Y3: [u64; LIMBS] = [
    41668, 23064, 49018, 65342, 36807, 58195, 42057, 26790, 39946, 45742, 24470, 59268, 31890,
    3594, 29580, 5613,
];

// x - x3 mod q
const TEST_X_MINUS_X3: [u64; LIMBS] = [
    11637, 27381, 33621, 26718, 44516, 20970, 21480, 48649, 17111, 6505, 16731, 7371, 34343, 45854,
    2443, 11614,
];

// x3 + 2*x
const TEST_X3_PLUS_2X: [u64; LIMBS] = [
    53205, 28039, 2241, 54210, 7336, 5767, 5801, 55672, 5509, 26648, 1115, 39813, 6658, 11795,
    17639, 774,
];

// y3 + y
const TEST_Y3_PLUS_Y: [u64; LIMBS] = [
    41670, 23064, 49018, 65342, 36807, 58195, 42057, 26790, 39946, 45742, 24470, 59268, 31890,
    3594, 29580, 5613,
];

// quotient for lambda * 4 = 3 + 3q
const K2: [u64; LIMBS] = [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

// quotient for lambda^2
const K3: [u64; LIMBS] = [
    20089, 55750, 20172, 29138, 495, 51904, 52209, 42296, 49588, 43224, 10038, 30637, 59927, 40619,
    27680, 6968,
];

// quotient for lambda*(x-x3)
const K4: [u64; LIMBS] = [
    57880, 4151, 57984, 20038, 619, 15728, 65262, 52870, 61985, 21262, 28932, 21912, 58525, 50774,
    34600, 8710,
];

// ============================================================
// One modular multiplication block.
//
// a*b = c + k*q
// ============================================================

const A_OFF: usize = 0;
const B_OFF: usize = A_OFF + LIMBS;
const C_OFF: usize = B_OFF + LIMBS;
const K_OFF: usize = C_OFF + LIMBS;

const AB_OFF: usize = K_OFF + LIMBS;

const AB_CARRY_OFF: usize = AB_OFF + WIDE_LIMBS;

const KQ_OFF: usize = AB_CARRY_OFF + 33;

const KQ_CARRY_OFF: usize = KQ_OFF + WIDE_LIMBS;

const ADD_CARRY_OFF: usize = KQ_CARRY_OFF + 33;

const RANGE_LIMBS: usize = 4 * LIMBS + 2 * WIDE_LIMBS;

const LIMB_BIT_OFF: usize = ADD_CARRY_OFF + 33;

const LIMB_BIT_COLS: usize = RANGE_LIMBS * LIMB_BITS;

const MUL_CARRIES: usize = 66;

const CARRY_BIT_OFF: usize = LIMB_BIT_OFF + LIMB_BIT_COLS;

const CARRY_BIT_COLS: usize = MUL_CARRIES * CARRY_BITS;

const BLOCK_WIDTH: usize = CARRY_BIT_OFF + CARRY_BIT_COLS;

// ============================================================
// Four modular multiplication blocks
// ============================================================

const MUL1_BASE: usize = 0;
const MUL2_BASE: usize = MUL1_BASE + BLOCK_WIDTH;
const MUL3_BASE: usize = MUL2_BASE + BLOCK_WIDTH;
const MUL4_BASE: usize = MUL3_BASE + BLOCK_WIDTH;

const MUL_BLOCKS_END: usize = MUL4_BASE + BLOCK_WIDTH;

// ============================================================
// Extra generic witnesses
//
// y
// x3
// y3
// two_x
// two_x_squared
//
// Each is a 16-limb field value.
// ============================================================

const Y_START: usize = MUL_BLOCKS_END;

const X3_START: usize = Y_START + LIMBS;

const Y3_START: usize = X3_START + LIMBS;

const TWO_X_START: usize = Y3_START + LIMBS;

const TWO_X2_START: usize = TWO_X_START + LIMBS;

const EXTRA_VALUES: usize = 5;
const EXTRA_LIMBS: usize = EXTRA_VALUES * LIMBS;

// ============================================================
// Modular-addition relations.
//
// We use seven:
//
// R0: y + y = 2y
// R1: x² + x² = 2x²
// R2: 2x² + x² = 3x²
// R3: x + x = 2x
// R4: x3 + 2x = block3.c
// R5: (x-x3) + x3 = x
// R6: y3 + y = block4.c
//
// For each relation:
//
//      a + b = c + r*q
//
// r is a boolean.
//
// We compute carries on both sides.
// ============================================================

const NUM_MODADDS: usize = 7;

const MODADD_CARRY_COUNT: usize = 17;

// r
// lhs carries[17]
// rhs carries[17]
const MODADD_WIDTH: usize = 1 + MODADD_CARRY_COUNT + MODADD_CARRY_COUNT;

const MODADD_START: usize = TWO_X2_START + LIMBS;

const MODADD_END: usize = MODADD_START + NUM_MODADDS * MODADD_WIDTH;

// ============================================================
// Range-check bits for the five extra field values
// ============================================================

const EXTRA_BIT_START: usize = MODADD_END;

const EXTRA_BIT_COLS: usize = EXTRA_LIMBS * LIMB_BITS;

const WIDTH: usize = EXTRA_BIT_START + EXTRA_BIT_COLS;

// ============================================================
// Modular multiplication AIR
// ============================================================

fn eval_modmul<AB: AirBuilder>(builder: &mut AB, row: &[AB::Var], base: usize) {
    let radix = AB::Expr::from_u64(RADIX);

    let a_start = base + A_OFF;

    let b_start = base + B_OFF;

    let c_start = base + C_OFF;

    let k_start = base + K_OFF;

    let ab_start = base + AB_OFF;

    let ab_carry_start = base + AB_CARRY_OFF;

    let kq_start = base + KQ_OFF;

    let kq_carry_start = base + KQ_CARRY_OFF;

    let add_carry_start = base + ADD_CARRY_OFF;

    let limb_bit_start = base + LIMB_BIT_OFF;

    let carry_bit_start = base + CARRY_BIT_OFF;

    // --------------------------------------------------------
    // a*b
    // --------------------------------------------------------

    let carry0: AB::Expr = row[ab_carry_start].into();

    builder.assert_zero(carry0);

    for out in 0..WIDE_LIMBS {
        let mut acc: AB::Expr = row[ab_carry_start + out].into();

        for i in 0..LIMBS {
            if out >= i {
                let j = out - i;

                if j < LIMBS {
                    let a_i: AB::Expr = row[a_start + i].into();

                    let b_j: AB::Expr = row[b_start + j].into();

                    acc = acc + a_i * b_j;
                }
            }
        }

        let result: AB::Expr = row[ab_start + out].into();

        let carry_out: AB::Expr = row[ab_carry_start + out + 1].into();

        builder.assert_zero(acc - result - radix.clone() * carry_out);
    }

    let final_ab: AB::Expr = row[ab_carry_start + WIDE_LIMBS].into();

    builder.assert_zero(final_ab);

    // --------------------------------------------------------
    // k*q
    // --------------------------------------------------------

    let kq_carry0: AB::Expr = row[kq_carry_start].into();

    builder.assert_zero(kq_carry0);

    for out in 0..WIDE_LIMBS {
        let mut acc: AB::Expr = row[kq_carry_start + out].into();

        for i in 0..LIMBS {
            if out >= i {
                let j = out - i;

                if j < LIMBS {
                    let k_i: AB::Expr = row[k_start + i].into();

                    let q_j = AB::Expr::from_u64(Q[j]);

                    acc = acc + k_i * q_j;
                }
            }
        }

        let result: AB::Expr = row[kq_start + out].into();

        let carry_out: AB::Expr = row[kq_carry_start + out + 1].into();

        builder.assert_zero(acc - result - radix.clone() * carry_out);
    }

    let final_kq: AB::Expr = row[kq_carry_start + WIDE_LIMBS].into();

    builder.assert_zero(final_kq);

    // --------------------------------------------------------
    // c+kq = ab
    // --------------------------------------------------------

    let add0: AB::Expr = row[add_carry_start].into();

    builder.assert_zero(add0);

    for i in 0..WIDE_LIMBS {
        let carry_in: AB::Expr = row[add_carry_start + i].into();

        let carry_out: AB::Expr = row[add_carry_start + i + 1].into();

        let kq_i: AB::Expr = row[kq_start + i].into();

        let ab_i: AB::Expr = row[ab_start + i].into();

        let c_i = if i < LIMBS {
            let x: AB::Expr = row[c_start + i].into();

            x
        } else {
            AB::Expr::ZERO
        };

        builder.assert_zero(c_i + kq_i + carry_in - ab_i - radix.clone() * carry_out.clone());

        builder.assert_zero(carry_out.clone() * (carry_out - AB::Expr::ONE));
    }

    let final_add: AB::Expr = row[add_carry_start + WIDE_LIMBS].into();

    builder.assert_zero(final_add);

    // --------------------------------------------------------
    // Limb range checks
    // --------------------------------------------------------

    for range_idx in 0..RANGE_LIMBS {
        let value_col = if range_idx < LIMBS {
            a_start + range_idx
        } else if range_idx < 2 * LIMBS {
            b_start + range_idx - LIMBS
        } else if range_idx < 3 * LIMBS {
            c_start + range_idx - 2 * LIMBS
        } else if range_idx < 4 * LIMBS {
            k_start + range_idx - 3 * LIMBS
        } else if range_idx < 4 * LIMBS + WIDE_LIMBS {
            ab_start + range_idx - 4 * LIMBS
        } else {
            kq_start + range_idx - 4 * LIMBS - WIDE_LIMBS
        };

        let value: AB::Expr = row[value_col].into();

        let mut reconstructed = AB::Expr::ZERO;

        let mut power = 1u64;

        for bit in 0..LIMB_BITS {
            let col = limb_bit_start + range_idx * LIMB_BITS + bit;

            let b: AB::Expr = row[col].into();

            builder.assert_zero(b.clone() * (b.clone() - AB::Expr::ONE));

            reconstructed = reconstructed + AB::Expr::from_u64(power) * b;

            power <<= 1;
        }

        builder.assert_zero(value - reconstructed);
    }

    // --------------------------------------------------------
    // Multiplication carry range checks
    // --------------------------------------------------------

    for idx in 0..MUL_CARRIES {
        let value_col = if idx < 33 {
            ab_carry_start + idx
        } else {
            kq_carry_start + idx - 33
        };

        let value: AB::Expr = row[value_col].into();

        let mut reconstructed = AB::Expr::ZERO;

        let mut power = 1u64;

        for bit in 0..CARRY_BITS {
            let col = carry_bit_start + idx * CARRY_BITS + bit;

            let b: AB::Expr = row[col].into();

            builder.assert_zero(b.clone() * (b.clone() - AB::Expr::ONE));

            reconstructed = reconstructed + AB::Expr::from_u64(power) * b;

            power <<= 1;
        }

        builder.assert_zero(value - reconstructed);
    }
}

// ============================================================
// Modular addition helper
//
// Enforce:
//
//     a + b = c + r*q
//
// where r in {0,1}.
//
// Two independent addition carry chains are used:
// lhs = a+b
// rhs = c+r*q
// ============================================================

fn eval_modadd<AB: AirBuilder>(
    builder: &mut AB,
    row: &[AB::Var],
    relation: usize,
    a_start: usize,
    b_start: usize,
    c_start: usize,
) {
    let base = MODADD_START + relation * MODADD_WIDTH;

    let r_col = base;

    let lhs_carry_start = base + 1;

    let rhs_carry_start = lhs_carry_start + MODADD_CARRY_COUNT;

    let r: AB::Expr = row[r_col].into();

    builder.assert_zero(r.clone() * (r.clone() - AB::Expr::ONE));

    let lhs0: AB::Expr = row[lhs_carry_start].into();

    let rhs0: AB::Expr = row[rhs_carry_start].into();

    builder.assert_zero(lhs0);
    builder.assert_zero(rhs0);

    let radix = AB::Expr::from_u64(RADIX);

    for i in 0..LIMBS {
        let a: AB::Expr = row[a_start + i].into();

        let b: AB::Expr = row[b_start + i].into();

        let c: AB::Expr = row[c_start + i].into();

        let lhs_in: AB::Expr = row[lhs_carry_start + i].into();

        let lhs_out: AB::Expr = row[lhs_carry_start + i + 1].into();

        let rhs_in: AB::Expr = row[rhs_carry_start + i].into();

        let rhs_out: AB::Expr = row[rhs_carry_start + i + 1].into();

        // carry bits are boolean

        builder.assert_zero(lhs_out.clone() * (lhs_out.clone() - AB::Expr::ONE));

        builder.assert_zero(rhs_out.clone() * (rhs_out.clone() - AB::Expr::ONE));

        let lhs_digit = a + b + lhs_in - radix.clone() * lhs_out;

        let rhs_digit = c + r.clone() * AB::Expr::from_u64(Q[i]) + rhs_in - radix.clone() * rhs_out;

        builder.assert_zero(lhs_digit - rhs_digit);
    }

    let lhs_final: AB::Expr = row[lhs_carry_start + LIMBS].into();

    let rhs_final: AB::Expr = row[rhs_carry_start + LIMBS].into();

    builder.assert_zero(lhs_final - rhs_final);
}

// ============================================================
// Generic doubling AIR
// ============================================================

struct GenericG1DoubleAir;

impl<F> BaseAir<F> for GenericG1DoubleAir {
    fn width(&self) -> usize {
        WIDTH
    }
}

impl<AB: AirBuilder> Air<AB> for GenericG1DoubleAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let row = main.current_slice();

        // ====================================================
        // Four modular multiplications
        // ====================================================

        eval_modmul(builder, row, MUL1_BASE);

        eval_modmul(builder, row, MUL2_BASE);

        eval_modmul(builder, row, MUL3_BASE);

        eval_modmul(builder, row, MUL4_BASE);

        // ====================================================
        // Dynamic linking
        //
        // Block 1:
        //
        //     x*x = x²
        //
        // Therefore block1.a = block1.b.
        // ====================================================

        for i in 0..LIMBS {
            let x1: AB::Expr = row[MUL1_BASE + A_OFF + i].into();

            let x2: AB::Expr = row[MUL1_BASE + B_OFF + i].into();

            builder.assert_zero(x1 - x2);
        }

        // ====================================================
        // Same lambda is used in blocks 2,3,4.
        //
        // block2.a = lambda
        // block3.a = lambda
        // block3.b = lambda
        // block4.a = lambda
        // ====================================================

        for i in 0..LIMBS {
            let lambda: AB::Expr = row[MUL2_BASE + A_OFF + i].into();

            let b3a: AB::Expr = row[MUL3_BASE + A_OFF + i].into();

            let b3b: AB::Expr = row[MUL3_BASE + B_OFF + i].into();

            let b4a: AB::Expr = row[MUL4_BASE + A_OFF + i].into();

            builder.assert_zero(b3a - lambda.clone());

            builder.assert_zero(b3b - lambda.clone());

            builder.assert_zero(b4a - lambda);
        }

        // ====================================================
        // Modular additions:
        // ====================================================

        // R0:
        //
        // y + y = 2y mod q
        //
        // 2y is block2.b

        eval_modadd(builder, row, 0, Y_START, Y_START, MUL2_BASE + B_OFF);

        // R1:
        //
        // x²+x² = 2x²
        //
        // x² = block1.c

        eval_modadd(
            builder,
            row,
            1,
            MUL1_BASE + C_OFF,
            MUL1_BASE + C_OFF,
            TWO_X2_START,
        );

        // R2:
        //
        // 2x²+x² = 3x²
        //
        // 3x² = block2.c

        eval_modadd(
            builder,
            row,
            2,
            TWO_X2_START,
            MUL1_BASE + C_OFF,
            MUL2_BASE + C_OFF,
        );

        // R3:
        //
        // x+x = 2x

        eval_modadd(
            builder,
            row,
            3,
            MUL1_BASE + A_OFF,
            MUL1_BASE + A_OFF,
            TWO_X_START,
        );

        // R4:
        //
        // x3 + 2x
        // =
        // lambda²
        //
        // lambda² is block3.c

        eval_modadd(builder, row, 4, X3_START, TWO_X_START, MUL3_BASE + C_OFF);

        // R5:
        //
        // (x-x3) + x3 = x
        //
        // x-x3 is block4.b

        eval_modadd(
            builder,
            row,
            5,
            MUL4_BASE + B_OFF,
            X3_START,
            MUL1_BASE + A_OFF,
        );

        // R6:
        //
        // y3 + y
        // =
        // lambda*(x-x3)
        //
        // RHS is block4.c

        eval_modadd(builder, row, 6, Y3_START, Y_START, MUL4_BASE + C_OFF);

        // ====================================================
        // Range-check extra field-value limbs.
        // ====================================================

        for extra_idx in 0..EXTRA_LIMBS {
            let value_col = Y_START + extra_idx;

            let value: AB::Expr = row[value_col].into();

            let mut reconstructed = AB::Expr::ZERO;

            let mut power = 1u64;

            for bit in 0..LIMB_BITS {
                let col = EXTRA_BIT_START + extra_idx * LIMB_BITS + bit;

                let b: AB::Expr = row[col].into();

                builder.assert_zero(b.clone() * (b.clone() - AB::Expr::ONE));

                reconstructed = reconstructed + AB::Expr::from_u64(power) * b;

                power <<= 1;
            }

            builder.assert_zero(value - reconstructed);
        }
    }
}

// ============================================================
// Witness helpers
// ============================================================

fn multiply_limbs(a: &[u64; LIMBS], b: &[u64; LIMBS]) -> ([u64; WIDE_LIMBS], [u64; 33]) {
    let mut product = [0u64; WIDE_LIMBS];

    let mut carries = [0u64; 33];

    for out in 0..WIDE_LIMBS {
        let mut acc = carries[out] as u128;

        for i in 0..LIMBS {
            if out >= i {
                let j = out - i;

                if j < LIMBS {
                    acc += (a[i] as u128) * (b[j] as u128);
                }
            }
        }

        product[out] = (acc % RADIX as u128) as u64;

        carries[out + 1] = (acc / RADIX as u128) as u64;
    }

    assert_eq!(carries[WIDE_LIMBS], 0);

    for carry in carries {
        assert!(carry < (1u64 << CARRY_BITS));
    }

    (product, carries)
}

fn add_c_plus_kq(
    c: &[u64; LIMBS],
    kq: &[u64; WIDE_LIMBS],
    expected: &[u64; WIDE_LIMBS],
) -> [u64; 33] {
    let mut carries = [0u64; 33];

    for i in 0..WIDE_LIMBS {
        let c_i = if i < LIMBS { c[i] } else { 0 };

        let acc = c_i as u128 + kq[i] as u128 + carries[i] as u128;

        let limb = (acc % RADIX as u128) as u64;

        carries[i + 1] = (acc / RADIX as u128) as u64;

        assert_eq!(limb, expected[i], "modmul relation failed at limb {}", i);
    }

    assert_eq!(carries[WIDE_LIMBS], 0);

    carries
}

fn push_bits(row: &mut Vec<Val>, value: u64, bits: usize) {
    assert!(value < (1u64 << bits));

    for i in 0..bits {
        row.push(Val::from_u64((value >> i) & 1));
    }
}

// ============================================================
// Build one modular multiplication block
// ============================================================

fn build_modmul_block(
    a: [u64; LIMBS],
    b: [u64; LIMBS],
    c: [u64; LIMBS],
    k: [u64; LIMBS],
) -> Vec<Val> {
    let (ab, ab_carries) = multiply_limbs(&a, &b);

    let (kq, kq_carries) = multiply_limbs(&k, &Q);

    let add_carries = add_c_plus_kq(&c, &kq, &ab);

    let mut row = Vec::with_capacity(BLOCK_WIDTH);

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
        push_bits(&mut row, x, LIMB_BITS);
    }

    for x in b {
        push_bits(&mut row, x, LIMB_BITS);
    }

    for x in c {
        push_bits(&mut row, x, LIMB_BITS);
    }

    for x in k {
        push_bits(&mut row, x, LIMB_BITS);
    }

    for x in ab {
        push_bits(&mut row, x, LIMB_BITS);
    }

    for x in kq {
        push_bits(&mut row, x, LIMB_BITS);
    }

    for x in ab_carries {
        push_bits(&mut row, x, CARRY_BITS);
    }

    for x in kq_carries {
        push_bits(&mut row, x, CARRY_BITS);
    }

    assert_eq!(row.len(), BLOCK_WIDTH);

    row
}

// ============================================================
// Addition witness helper
//
// Computes carry chains for:
//
//     a+b = c+r*q
// ============================================================

fn addition_carries(a: &[u64; LIMBS], b: &[u64; LIMBS]) -> [u64; MODADD_CARRY_COUNT] {
    let mut carry = [0u64; MODADD_CARRY_COUNT];

    for i in 0..LIMBS {
        let v = a[i] as u128 + b[i] as u128 + carry[i] as u128;

        carry[i + 1] = (v / RADIX as u128) as u64;

        assert!(carry[i + 1] <= 1);
    }

    carry
}

fn addition_rhs_carries(c: &[u64; LIMBS], reduction: u64) -> [u64; MODADD_CARRY_COUNT] {
    let mut carry = [0u64; MODADD_CARRY_COUNT];

    for i in 0..LIMBS {
        let v = c[i] as u128 + reduction as u128 * Q[i] as u128 + carry[i] as u128;

        carry[i + 1] = (v / RADIX as u128) as u64;

        assert!(carry[i + 1] <= 1);
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
    assert!(reduction <= 1);

    let lhs = addition_carries(a, b);

    let rhs = addition_rhs_carries(c, reduction);

    // Verify relation limb-by-limb here too.

    for i in 0..LIMBS {
        let lhs_digit =
            (a[i] as i128 + b[i] as i128 + lhs[i] as i128 - RADIX as i128 * lhs[i + 1] as i128);

        let rhs_digit = (c[i] as i128 + reduction as i128 * Q[i] as i128 + rhs[i] as i128
            - RADIX as i128 * rhs[i + 1] as i128);

        assert_eq!(
            lhs_digit, rhs_digit,
            "bad modular-add witness at limb {}",
            i
        );
    }

    assert_eq!(lhs[LIMBS], rhs[LIMBS]);

    row.push(Val::from_u64(reduction));

    for x in lhs {
        row.push(Val::from_u64(x));
    }

    for x in rhs {
        row.push(Val::from_u64(x));
    }
}

// ============================================================
// Complete generic-doubling trace
// ============================================================

fn make_trace(height: usize) -> RowMajorMatrix<Val> {
    // --------------------------------------------------------
    // This is merely our test witness.
    //
    // AIR does NOT constrain x,y,x3,y3 to these constants.
    // --------------------------------------------------------

    let x = ONE;

    let y = TWO;

    let x_squared = ONE;

    let two_x = TWO;

    let two_x_squared = TWO;

    let three_x_squared = THREE;

    let two_y = FOUR;

    let lambda = TEST_LAMBDA;

    let x3 = TEST_X3;

    let y3 = TEST_Y3;

    let x_minus_x3 = TEST_X_MINUS_X3;

    let x3_plus_2x = TEST_X3_PLUS_2X;

    let y3_plus_y = TEST_Y3_PLUS_Y;

    // --------------------------------------------------------
    // Four modular multiplications
    // --------------------------------------------------------

    // M1:
    // x*x = x²

    let block1 = build_modmul_block(x, x, x_squared, ZERO);

    // M2:
    //
    // lambda*(2y)
    // =
    // 3x²

    let block2 = build_modmul_block(lambda, two_y, three_x_squared, K2);

    // M3:
    //
    // lambda²
    // =
    // x3 + 2x

    let block3 = build_modmul_block(lambda, lambda, x3_plus_2x, K3);

    // M4:
    //
    // lambda*(x-x3)
    // =
    // y3+y

    let block4 = build_modmul_block(lambda, x_minus_x3, y3_plus_y, K4);

    let mut one_row = Vec::with_capacity(WIDTH);

    one_row.extend_from_slice(&block1);

    one_row.extend_from_slice(&block2);

    one_row.extend_from_slice(&block3);

    one_row.extend_from_slice(&block4);

    // --------------------------------------------------------
    // Extra witnesses:
    //
    // y
    // x3
    // y3
    // 2x
    // 2x²
    // --------------------------------------------------------

    for v in y {
        one_row.push(Val::from_u64(v));
    }

    for v in x3 {
        one_row.push(Val::from_u64(v));
    }

    for v in y3 {
        one_row.push(Val::from_u64(v));
    }

    for v in two_x {
        one_row.push(Val::from_u64(v));
    }

    for v in two_x_squared {
        one_row.push(Val::from_u64(v));
    }

    // --------------------------------------------------------
    // Seven modular-addition witnesses
    // --------------------------------------------------------

    // R0
    // y+y = 2y
    push_modadd_witness(&mut one_row, &y, &y, &two_y, 0);

    // R1
    // x²+x² = 2x²
    push_modadd_witness(&mut one_row, &x_squared, &x_squared, &two_x_squared, 0);

    // R2
    // 2x²+x² = 3x²
    push_modadd_witness(
        &mut one_row,
        &two_x_squared,
        &x_squared,
        &three_x_squared,
        0,
    );

    // R3
    // x+x = 2x
    push_modadd_witness(&mut one_row, &x, &x, &two_x, 0);

    // R4
    // x3+2x = x3+2
    push_modadd_witness(&mut one_row, &x3, &two_x, &x3_plus_2x, 0);

    // R5
    //
    // (x-x3)+x3 = x + q
    //
    // so reduction bit = 1

    push_modadd_witness(&mut one_row, &x_minus_x3, &x3, &x, 1);

    // R6
    // y3+y

    push_modadd_witness(&mut one_row, &y3, &y, &y3_plus_y, 0);

    // --------------------------------------------------------
    // Extra witness range-check bits
    // --------------------------------------------------------

    for value in y {
        push_bits(&mut one_row, value, LIMB_BITS);
    }

    for value in x3 {
        push_bits(&mut one_row, value, LIMB_BITS);
    }

    for value in y3 {
        push_bits(&mut one_row, value, LIMB_BITS);
    }

    for value in two_x {
        push_bits(&mut one_row, value, LIMB_BITS);
    }

    for value in two_x_squared {
        push_bits(&mut one_row, value, LIMB_BITS);
    }

    assert_eq!(
        one_row.len(),
        WIDTH,
        "trace width mismatch: {} != {}",
        one_row.len(),
        WIDTH
    );

    let mut values = Vec::with_capacity(WIDTH * height);

    for _ in 0..height {
        values.extend_from_slice(&one_row);
    }

    RowMajorMatrix::new(values, WIDTH)
}

// ============================================================
// Main
// ============================================================

fn main() {
    println!("==============================================");

    println!(" Plonky3 Generic BN254 G1 Doubling");

    println!("==============================================");

    println!();

    println!("Native proving field : Goldilocks");

    println!("Target curve         : BN254 G1");

    println!();

    println!("AIR proves:");

    println!("  lambda*(2y) = 3*x^2");

    println!("  lambda^2 = x3 + 2*x");

    println!("  lambda*(x-x3) = y3 + y");

    println!();

    println!("Test witness:");

    println!("  P = (1,2)");

    println!();

    println!("Important:");

    println!("  x,y,lambda,x3,y3 are linked dynamically.");

    println!("  They are not fixed to constants by the AIR.");

    println!();

    let trace_height = 8usize;

    let trace = make_trace(trace_height);

    // ========================================================
    // Merkle / FRI configuration
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

    type Config = StarkConfig<Pcs, Challenge, Challenger>;

    let config = Config::new(pcs, challenger);

    // ========================================================
    // Prove
    // ========================================================

    let prove_start = Instant::now();

    let proof = prove(&config, &GenericG1DoubleAir, trace, &[]).expect("proof generation failed");

    let prove_time = prove_start.elapsed();

    // ========================================================
    // Serialize
    // ========================================================

    let proof_bytes = postcard::to_allocvec(&proof).expect("proof serialization failed");

    // ========================================================
    // Verify
    // ========================================================

    let verify_start = Instant::now();

    verify(&config, &GenericG1DoubleAir, &proof, &[]).expect("verification failed");

    let verify_time = verify_start.elapsed();

    // ========================================================
    // Metrics
    // ========================================================

    const MODMUL_CONSTRAINTS: usize = 3696;

    // block1 a=b
    // = 16
    //
    // lambda links:
    // block3.a
    // block3.b
    // block4.a
    //
    // = 3*16 = 48
    //
    // total = 64

    const LINK_CONSTRAINTS: usize = 64;

    // Extra 5*16 limbs.
    //
    // Each:
    // 16 boolean constraints
    // + 1 reconstruction
    //
    // 80*17 = 1360

    const EXTRA_RANGE_CONSTRAINTS: usize = EXTRA_LIMBS * (LIMB_BITS + 1);

    // Each modular-add relation:
    //
    // 1 reduction-bit boolean
    // 2 starting carry-zero constraints
    // 16 limb equations
    // 32 carry boolean constraints
    // 1 final carry equality
    //
    // = 52 constraints

    const MODADD_CONSTRAINTS_PER_RELATION: usize = 1 + 2 + 16 + 32 + 1;

    const MODADD_CONSTRAINTS: usize = NUM_MODADDS * MODADD_CONSTRAINTS_PER_RELATION;

    let total_constraints =
        4 * MODMUL_CONSTRAINTS + LINK_CONSTRAINTS + EXTRA_RANGE_CONSTRAINTS + MODADD_CONSTRAINTS;

    println!("==============================================");

    println!(" Metrics");

    println!("==============================================");

    println!();

    println!("Modular multiplications    : 4");

    println!("Modular additions         : {}", NUM_MODADDS);

    println!("Constraints / modmul       : {}", MODMUL_CONSTRAINTS);

    println!("Dynamic link constraints  : {}", LINK_CONSTRAINTS);

    println!("Extra range constraints   : {}", EXTRA_RANGE_CONSTRAINTS);

    println!("Modadd constraints        : {}", MODADD_CONSTRAINTS);

    println!("AIR constraint expressions : {}", total_constraints);

    println!("Maximum constraint degree  : 2");

    println!("Trace width                : {} columns", WIDTH);

    println!("Trace height               : {} rows", trace_height);

    println!("Trace cells                : {}", WIDTH * trace_height);

    println!(
        "Row-wise evaluations       : {}",
        total_constraints * trace_height
    );

    println!("Proof size                 : {} bytes", proof_bytes.len());

    println!("Proving time               : {:?}", prove_time);

    println!("Verification time          : {:?}", verify_time);

    println!();

    println!("Generic BN254 G1 doubling proof verified!");
}
