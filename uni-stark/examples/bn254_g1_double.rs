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
// Non-native BN254 representation
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
// Fixed doubling test:
//
// P = (1, 2)
// 2P = (X3, Y3)
// ============================================================

const ZERO: [u64; LIMBS] = [0; LIMBS];

const ONE: [u64; LIMBS] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

const TWO: [u64; LIMBS] = [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

const THREE: [u64; LIMBS] = [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

const FOUR: [u64; LIMBS] = [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

// lambda = 3 / 4 mod q
const LAMBDA: [u64; LIMBS] = [
    48630, 8797, 26897, 60696, 22505, 3669, 4077, 12705, 582, 57633, 13384, 19004, 14367, 10469,
    15062, 9291,
];

// x-coordinate of 2*(1,2)
const X3: [u64; LIMBS] = [
    53203, 28039, 2241, 54210, 7336, 5767, 5801, 55672, 5509, 26648, 1115, 39813, 6658, 11795,
    17639, 774,
];

// y-coordinate of 2*(1,2)
const Y3: [u64; LIMBS] = [
    41668, 23064, 49018, 65342, 36807, 58195, 42057, 26790, 39946, 45742, 24470, 59268, 31890,
    3594, 29580, 5613,
];

// x - x3 mod q
const X_MINUS_X3: [u64; LIMBS] = [
    11637, 27381, 33621, 26718, 44516, 20970, 21480, 48649, 17111, 6505, 16731, 7371, 34343, 45854,
    2443, 11614,
];

// x3 + 2*x = x3 + 2
const X3_PLUS_2: [u64; LIMBS] = [
    53205, 28039, 2241, 54210, 7336, 5767, 5801, 55672, 5509, 26648, 1115, 39813, 6658, 11795,
    17639, 774,
];

// y3 + y = y3 + 2
const Y3_PLUS_2: [u64; LIMBS] = [
    41670, 23064, 49018, 65342, 36807, 58195, 42057, 26790, 39946, 45742, 24470, 59268, 31890,
    3594, 29580, 5613,
];

// Quotient for:
// lambda * 4 = 3 + K2*q
const K2: [u64; LIMBS] = [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

// Quotient for:
// lambda^2 = (x3 + 2) + K3*q
const K3: [u64; LIMBS] = [
    20089, 55750, 20172, 29138, 495, 51904, 52209, 42296, 49588, 43224, 10038, 30637, 59927, 40619,
    27680, 6968,
];

// Quotient for:
// lambda*(x-x3) = (y3+2) + K4*q
const K4: [u64; LIMBS] = [
    57880, 4151, 57984, 20038, 619, 15728, 65262, 52870, 61985, 21262, 28932, 21912, 58525, 50774,
    34600, 8710,
];

// ============================================================
// One modular multiplication block
//
// proves:
//
//     a*b = c + k*q
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

const WIDTH: usize = 4 * BLOCK_WIDTH;

// ============================================================
// Modular multiplication AIR helper
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

    // ========================================================
    // 1. a*b
    // ========================================================

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

    let final_ab_carry: AB::Expr = row[ab_carry_start + WIDE_LIMBS].into();

    builder.assert_zero(final_ab_carry);

    // ========================================================
    // 2. k*q
    // ========================================================

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

    let final_kq_carry: AB::Expr = row[kq_carry_start + WIDE_LIMBS].into();

    builder.assert_zero(final_kq_carry);

    // ========================================================
    // 3. c + k*q = a*b
    // ========================================================

    let add_carry0: AB::Expr = row[add_carry_start].into();

    builder.assert_zero(add_carry0);

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

    let final_add_carry: AB::Expr = row[add_carry_start + WIDE_LIMBS].into();

    builder.assert_zero(final_add_carry);

    // ========================================================
    // 4. Range checks for 16-bit limbs
    // ========================================================

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
            let bit_col = limb_bit_start + range_idx * LIMB_BITS + bit;

            let b: AB::Expr = row[bit_col].into();

            builder.assert_zero(b.clone() * (b.clone() - AB::Expr::ONE));

            reconstructed = reconstructed + AB::Expr::from_u64(power) * b;

            power <<= 1;
        }

        builder.assert_zero(value - reconstructed);
    }

    // ========================================================
    // 5. Carry range checks
    // ========================================================

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
            let bit_col = carry_bit_start + idx * CARRY_BITS + bit;

            let b: AB::Expr = row[bit_col].into();

            builder.assert_zero(b.clone() * (b.clone() - AB::Expr::ONE));

            reconstructed = reconstructed + AB::Expr::from_u64(power) * b;

            power <<= 1;
        }

        builder.assert_zero(value - reconstructed);
    }
}

// ============================================================
// G1 point-doubling AIR
// ============================================================

struct G1DoubleAir;

impl<F> BaseAir<F> for G1DoubleAir {
    fn width(&self) -> usize {
        WIDTH
    }
}

impl<AB: AirBuilder> Air<AB> for G1DoubleAir {
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
        // Block 1:
        //
        // x*x = x^2
        //
        // x = 1
        // x^2 = 1
        // ====================================================

        for i in 0..LIMBS {
            let a: AB::Expr = row[MUL1_BASE + A_OFF + i].into();

            let b: AB::Expr = row[MUL1_BASE + B_OFF + i].into();

            let c: AB::Expr = row[MUL1_BASE + C_OFF + i].into();

            builder.assert_zero(a - AB::Expr::from_u64(ONE[i]));

            builder.assert_zero(b - AB::Expr::from_u64(ONE[i]));

            builder.assert_zero(c - AB::Expr::from_u64(ONE[i]));
        }

        // ====================================================
        // Block 2:
        //
        // lambda*(2y) = 3*x^2
        //
        // y=2, x=1:
        //
        // lambda*4 = 3 mod q
        // ====================================================

        for i in 0..LIMBS {
            let a: AB::Expr = row[MUL2_BASE + A_OFF + i].into();

            let b: AB::Expr = row[MUL2_BASE + B_OFF + i].into();

            let c: AB::Expr = row[MUL2_BASE + C_OFF + i].into();

            let k: AB::Expr = row[MUL2_BASE + K_OFF + i].into();

            builder.assert_zero(a - AB::Expr::from_u64(LAMBDA[i]));

            builder.assert_zero(b - AB::Expr::from_u64(FOUR[i]));

            builder.assert_zero(c - AB::Expr::from_u64(THREE[i]));

            builder.assert_zero(k - AB::Expr::from_u64(K2[i]));
        }

        // ====================================================
        // Block 3:
        //
        // lambda^2 = x3 + 2*x
        //
        // x=1:
        //
        // lambda^2 = x3 + 2
        // ====================================================

        for i in 0..LIMBS {
            let a: AB::Expr = row[MUL3_BASE + A_OFF + i].into();

            let b: AB::Expr = row[MUL3_BASE + B_OFF + i].into();

            let c: AB::Expr = row[MUL3_BASE + C_OFF + i].into();

            let k: AB::Expr = row[MUL3_BASE + K_OFF + i].into();

            builder.assert_zero(a - AB::Expr::from_u64(LAMBDA[i]));

            builder.assert_zero(b - AB::Expr::from_u64(LAMBDA[i]));

            builder.assert_zero(c - AB::Expr::from_u64(X3_PLUS_2[i]));

            builder.assert_zero(k - AB::Expr::from_u64(K3[i]));
        }

        // ====================================================
        // Block 4:
        //
        // lambda*(x-x3)
        // =
        // y3 + y
        //
        // y=2
        // ====================================================

        for i in 0..LIMBS {
            let a: AB::Expr = row[MUL4_BASE + A_OFF + i].into();

            let b: AB::Expr = row[MUL4_BASE + B_OFF + i].into();

            let c: AB::Expr = row[MUL4_BASE + C_OFF + i].into();

            let k: AB::Expr = row[MUL4_BASE + K_OFF + i].into();

            builder.assert_zero(a - AB::Expr::from_u64(LAMBDA[i]));

            builder.assert_zero(b - AB::Expr::from_u64(X_MINUS_X3[i]));

            builder.assert_zero(c - AB::Expr::from_u64(Y3_PLUS_2[i]));

            builder.assert_zero(k - AB::Expr::from_u64(K4[i]));
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

        assert_eq!(limb, expected[i], "modular relation failed at limb {}", i);

        assert!(carries[i + 1] <= 1);
    }

    assert_eq!(carries[WIDE_LIMBS], 0);

    carries
}

fn push_bits(row: &mut Vec<Val>, value: u64, bits: usize) {
    assert!(value < (1u64 << bits));

    for i in 0..bits {
        let bit = (value >> i) & 1;

        row.push(Val::from_u64(bit));
    }
}

// ============================================================
// Build one modular multiplication witness block
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

    // a
    for x in a {
        row.push(Val::from_u64(x));
    }

    // b
    for x in b {
        row.push(Val::from_u64(x));
    }

    // c
    for x in c {
        row.push(Val::from_u64(x));
    }

    // k
    for x in k {
        row.push(Val::from_u64(x));
    }

    // ab
    for x in ab {
        row.push(Val::from_u64(x));
    }

    // ab carries
    for x in ab_carries {
        row.push(Val::from_u64(x));
    }

    // kq
    for x in kq {
        row.push(Val::from_u64(x));
    }

    // kq carries
    for x in kq_carries {
        row.push(Val::from_u64(x));
    }

    // addition carries
    for x in add_carries {
        row.push(Val::from_u64(x));
    }

    // ========================================================
    // Limb range-check bits
    // ========================================================

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

    // ========================================================
    // Multiplication carry range-check bits
    // ========================================================

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
// Complete trace
// ============================================================

fn make_trace(height: usize) -> RowMajorMatrix<Val> {
    // ========================================================
    // Block 1:
    //
    // x*x = x^2
    //
    // 1*1 = 1
    // ========================================================

    let block1 = build_modmul_block(ONE, ONE, ONE, ZERO);

    // ========================================================
    // Block 2:
    //
    // lambda*(2y)
    // =
    // 3*x^2
    //
    // lambda*4 = 3 mod q
    // ========================================================

    let block2 = build_modmul_block(LAMBDA, FOUR, THREE, K2);

    // ========================================================
    // Block 3:
    //
    // lambda^2
    // =
    // x3 + 2*x
    // ========================================================

    let block3 = build_modmul_block(LAMBDA, LAMBDA, X3_PLUS_2, K3);

    // ========================================================
    // Block 4:
    //
    // lambda*(x-x3)
    // =
    // y3 + y
    // ========================================================

    let block4 = build_modmul_block(LAMBDA, X_MINUS_X3, Y3_PLUS_2, K4);

    let mut one_row = Vec::with_capacity(WIDTH);

    one_row.extend_from_slice(&block1);

    one_row.extend_from_slice(&block2);

    one_row.extend_from_slice(&block3);

    one_row.extend_from_slice(&block4);

    assert_eq!(one_row.len(), WIDTH);

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

    println!(" Plonky3 BN254 G1 Point Doubling");

    println!("==============================================");

    println!();

    println!("Native proving field : Goldilocks");

    println!("Target curve         : BN254 G1");

    println!();

    println!("Input point:");

    println!("  P = (1,2)");

    println!();

    println!("Doubling equations:");

    println!("  lambda*(2y) = 3*x^2");

    println!("  x3 = lambda^2 - 2*x");

    println!("  y3 = lambda*(x-x3) - y");

    println!();

    println!("Expected doubled point:");

    println!("  x3 = 1368015179489954701390400359078579693043519447331113978918064868415326638035");

    println!("  y3 = 9918110051302171585080402603319702774565515993150576347155970296011118125764");

    println!();

    // Keep compiler from warning that X3/Y3 are unused.
    let _ = X3;

    let _ = Y3;

    let _ = TWO;

    let trace_height = 8usize;

    let trace = make_trace(trace_height);

    // ========================================================
    // Hash / Merkle
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

    // ========================================================
    // FRI
    // ========================================================

    type Dft = Radix2DitParallel<Val>;

    let dft = Dft::default();

    let fri_params = FriParameters::new_benchmark(challenge_mmcs);

    type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;

    let pcs = Pcs::new(dft, val_mmcs, fri_params);

    // ========================================================
    // Challenger
    // ========================================================

    type Challenger = SerializingChallenger64<Val, HashChallenger<u8, ByteHash, 32>>;

    let challenger = Challenger::from_hasher(vec![], byte_hash);

    type Config = StarkConfig<Pcs, Challenge, Challenger>;

    let config = Config::new(pcs, challenger);

    // ========================================================
    // Prove
    // ========================================================

    let prove_start = Instant::now();

    let proof = prove(&config, &G1DoubleAir, trace, &[]).expect("proof generation failed");

    let prove_time = prove_start.elapsed();

    // ========================================================
    // Serialize
    // ========================================================

    let proof_bytes = postcard::to_allocvec(&proof).expect("proof serialization failed");

    // ========================================================
    // Verify
    // ========================================================

    let verify_start = Instant::now();

    verify(&config, &G1DoubleAir, &proof, &[]).expect("verification failed");

    let verify_time = verify_start.elapsed();

    // ========================================================
    // Metrics
    // ========================================================

    const MODMUL_CONSTRAINTS: usize = 3696;

    const DOUBLING_LINK_CONSTRAINTS: usize = 240;

    let total_constraints = 4 * MODMUL_CONSTRAINTS + DOUBLING_LINK_CONSTRAINTS;

    println!("==============================================");

    println!(" Metrics");

    println!("==============================================");

    println!();

    println!("Modular multiplications    : 4");

    println!("Constraints / modmul       : {}", MODMUL_CONSTRAINTS);

    println!("Doubling/link constraints : {}", DOUBLING_LINK_CONSTRAINTS);

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

    println!("BN254 G1 point-doubling proof verified!");
}
