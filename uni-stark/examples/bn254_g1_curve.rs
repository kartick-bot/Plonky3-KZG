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
// BN254 base-field modulus
// ============================================================

const Q: [u64; LIMBS] = [
    64839, 55420, 35862, 15392, 51853, 26737, 27281, 38785, 22621, 33153, 17846, 47184, 41001,
    57649, 20082, 12388,
];

// ============================================================
// One modular multiplication block
//
// proves:
//
//     a*b = c + k*q
//
// Layout is identical to our previous working experiment.
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
// Three modular multiplication blocks
//
// block 0: x*x   = x^2
// block 1: x^2*x = x^3
// block 2: y*y   = y^2
// ============================================================

const MUL1_BASE: usize = 0;

const MUL2_BASE: usize = MUL1_BASE + BLOCK_WIDTH;

const MUL3_BASE: usize = MUL2_BASE + BLOCK_WIDTH;

const WIDTH: usize = 3 * BLOCK_WIDTH;

// ============================================================
// Helper: enforce one modular multiplication block
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

    let ab_final: AB::Expr = row[ab_carry_start + WIDE_LIMBS].into();

    builder.assert_zero(ab_final);

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

    let kq_final: AB::Expr = row[kq_carry_start + WIDE_LIMBS].into();

    builder.assert_zero(kq_final);

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
    // 4. 16-bit range checks
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
// G1 AIR
// ============================================================

struct G1CurveAir;

impl<F> BaseAir<F> for G1CurveAir {
    fn width(&self) -> usize {
        WIDTH
    }
}

impl<AB: AirBuilder> Air<AB> for G1CurveAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let row = main.current_slice();

        // ====================================================
        // Modular multiplications
        // ====================================================

        eval_modmul(builder, row, MUL1_BASE);

        eval_modmul(builder, row, MUL2_BASE);

        eval_modmul(builder, row, MUL3_BASE);

        // ====================================================
        // Link block 1:
        //
        // x*x = t1
        //
        // Force x = 1.
        // ====================================================

        for i in 0..LIMBS {
            let a: AB::Expr = row[MUL1_BASE + A_OFF + i].into();

            let b: AB::Expr = row[MUL1_BASE + B_OFF + i].into();

            let expected = if i == 0 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            };

            builder.assert_zero(a - expected.clone());

            builder.assert_zero(b - expected);
        }

        // ====================================================
        // Link block 2:
        //
        // t1 * x = t2
        // ====================================================

        for i in 0..LIMBS {
            let t1: AB::Expr = row[MUL1_BASE + C_OFF + i].into();

            let mul2_a: AB::Expr = row[MUL2_BASE + A_OFF + i].into();

            builder.assert_zero(mul2_a - t1);

            let mul2_b: AB::Expr = row[MUL2_BASE + B_OFF + i].into();

            let expected_x = if i == 0 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            };

            builder.assert_zero(mul2_b - expected_x);
        }

        // ====================================================
        // Block 3:
        //
        // y*y = t3
        //
        // Force y = 2.
        // ====================================================

        for i in 0..LIMBS {
            let y1: AB::Expr = row[MUL3_BASE + A_OFF + i].into();

            let y2: AB::Expr = row[MUL3_BASE + B_OFF + i].into();

            let expected = if i == 0 {
                AB::Expr::from_u64(2)
            } else {
                AB::Expr::ZERO
            };

            builder.assert_zero(y1 - expected.clone());

            builder.assert_zero(y2 - expected);
        }

        // ====================================================
        // Final curve equation:
        //
        // y^2 = x^3 + 3
        //
        // For point (1,2):
        //
        // 4 = 1 + 3
        // ====================================================

        for i in 0..LIMBS {
            let x3: AB::Expr = row[MUL2_BASE + C_OFF + i].into();

            let y2: AB::Expr = row[MUL3_BASE + C_OFF + i].into();

            if i == 0 {
                builder.assert_zero(y2 - x3 - AB::Expr::from_u64(3));
            } else {
                builder.assert_zero(y2 - x3);
            }
        }
    }
}

// ============================================================
// Witness helpers
// ============================================================

fn small_limbs(value: u64) -> [u64; LIMBS] {
    let mut out = [0u64; LIMBS];

    out[0] = value;

    out
}

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

        assert_eq!(limb, expected[i]);
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
// Build one mod-mul witness block
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

    // a,b,c,k bits
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

    // ab bits
    for x in ab {
        push_bits(&mut row, x, LIMB_BITS);
    }

    // kq bits
    for x in kq {
        push_bits(&mut row, x, LIMB_BITS);
    }

    // multiplication carry bits
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
// Build complete G1 trace
// ============================================================

fn make_trace(height: usize) -> RowMajorMatrix<Val> {
    let x = small_limbs(1);

    let y = small_limbs(2);

    let t1 = small_limbs(1);

    let t2 = small_limbs(1);

    let t3 = small_limbs(4);

    let zero = small_limbs(0);

    // x*x = 1
    let block1 = build_modmul_block(x, x, t1, zero);

    // t1*x = 1
    let block2 = build_modmul_block(t1, x, t2, zero);

    // y*y = 4
    let block3 = build_modmul_block(y, y, t3, zero);

    let mut one_row = Vec::with_capacity(WIDTH);

    one_row.extend_from_slice(&block1);

    one_row.extend_from_slice(&block2);

    one_row.extend_from_slice(&block3);

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
    println!(" Plonky3 BN254 G1 Curve Membership");
    println!("==============================================");
    println!();

    println!("Native proving field : Goldilocks");

    println!("Target curve         : BN254 G1");

    println!("Curve equation       : y^2 = x^3 + 3");

    println!();

    println!("Point:");

    println!("  x = 1");

    println!("  y = 2");

    println!();

    println!("Computation:");

    println!("  x^2 = 1");

    println!("  x^3 = 1");

    println!("  y^2 = 4");

    println!("  4 = 1 + 3");

    println!();

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

    type Challenger = SerializingChallenger64<Val, HashChallenger<u8, ByteHash, 32>>;

    let challenger = Challenger::from_hasher(vec![], byte_hash);

    type Config = StarkConfig<Pcs, Challenge, Challenger>;

    let config = Config::new(pcs, challenger);

    // ========================================================
    // Prove
    // ========================================================

    let prove_start = Instant::now();

    let proof = prove(&config, &G1CurveAir, trace, &[]).expect("proof generation failed");

    let prove_time = prove_start.elapsed();

    // ========================================================
    // Proof size
    // ========================================================

    let proof_bytes = postcard::to_allocvec(&proof).expect("serialization failed");

    // ========================================================
    // Verify
    // ========================================================

    let verify_start = Instant::now();

    verify(&config, &G1CurveAir, &proof, &[]).expect("verification failed");

    let verify_time = verify_start.elapsed();

    // ========================================================
    // Metrics
    // ========================================================

    const MODMUL_CONSTRAINTS: usize = 3696;

    // Link constraints:
    //
    // block1 x = 1:
    //   16 + 16 = 32
    //
    // block2:
    //   t1 link = 16
    //   x link  = 16
    //
    // block3 y = 2:
    //   16 + 16 = 32
    //
    // final curve equality:
    //   16
    //
    // Total:
    //
    // 32 + 32 + 32 + 16 = 112

    const LINK_CONSTRAINTS: usize = 112;

    let total_constraints = 3 * MODMUL_CONSTRAINTS + LINK_CONSTRAINTS;

    println!("==============================================");

    println!(" Metrics");

    println!("==============================================");

    println!();

    println!("Modular multiplications    : 3");

    println!("Constraints / modmul       : {}", MODMUL_CONSTRAINTS);

    println!("Link/curve constraints     : {}", LINK_CONSTRAINTS);

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

    println!("BN254 G1 curve-membership proof verified!");
}
