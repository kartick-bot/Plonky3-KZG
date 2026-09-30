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
// BN254 base-field modulus q
//
// q =
// 21888242871839275222246405745257275088696311157297823662689037894645226208583
//
// little-endian 16-bit limbs
// ============================================================

const Q: [u64; LIMBS] = [
    64839, 55420, 35862, 15392,
    51853, 26737, 27281, 38785,
    22621, 33153, 17846, 47184,
    41001, 57649, 20082, 12388,
];

// ============================================================
// Trace layout
// ============================================================

// a
const A_START: usize = 0;

// b
const B_START: usize = A_START + LIMBS;

// c
const C_START: usize = B_START + LIMBS;

// quotient k
const K_START: usize = C_START + LIMBS;

// ab
const AB_START: usize = K_START + LIMBS;

// carries for ab
const AB_CARRY_START: usize =
    AB_START + WIDE_LIMBS;

// kq
const KQ_START: usize =
    AB_CARRY_START + 33;

// carries for kq
const KQ_CARRY_START: usize =
    KQ_START + WIDE_LIMBS;

// carries for c + kq = ab
const ADD_CARRY_START: usize =
    KQ_CARRY_START + 33;

// ============================================================
// Range-check columns
//
// Range-check:
// a,b,c,k               = 64 limbs
// ab product             = 32
// kq product             = 32
//
// Total = 128 16-bit values.
// ============================================================

const RANGE_LIMBS: usize =
    4 * LIMBS + 2 * WIDE_LIMBS;

const LIMB_BIT_START: usize =
    ADD_CARRY_START + 33;

const LIMB_BIT_COLS: usize =
    RANGE_LIMBS * LIMB_BITS;

// Two multiplication carry arrays:
// 33 + 33 = 66 carries.
const MUL_CARRIES: usize = 66;

const CARRY_BIT_START: usize =
    LIMB_BIT_START + LIMB_BIT_COLS;

const CARRY_BIT_COLS: usize =
    MUL_CARRIES * CARRY_BITS;

const WIDTH: usize =
    CARRY_BIT_START + CARRY_BIT_COLS;


// ============================================================
// AIR
// ============================================================

struct Bn254ModMulAir;

impl<F> BaseAir<F> for Bn254ModMulAir {
    fn width(&self) -> usize {
        WIDTH
    }
}

impl<AB: AirBuilder> Air<AB> for Bn254ModMulAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.current_slice();

        let radix = AB::Expr::from_u64(RADIX);

        // ====================================================
        // 1. Prove ab
        // ====================================================

        let carry0: AB::Expr =
            row[AB_CARRY_START].into();

        builder.assert_zero(carry0);

        for out in 0..WIDE_LIMBS {
            let mut acc: AB::Expr =
                row[AB_CARRY_START + out].into();

            for i in 0..LIMBS {
                if out >= i {
                    let j = out - i;

                    if j < LIMBS {
                        let a_i: AB::Expr =
                            row[A_START + i].into();

                        let b_j: AB::Expr =
                            row[B_START + j].into();

                        acc = acc + a_i * b_j;
                    }
                }
            }

            let result: AB::Expr =
                row[AB_START + out].into();

            let next_carry: AB::Expr =
                row[AB_CARRY_START + out + 1].into();

            builder.assert_zero(
                acc
                    - result
                    - radix.clone() * next_carry
            );
        }

        let final_ab_carry: AB::Expr =
            row[AB_CARRY_START + WIDE_LIMBS].into();

        builder.assert_zero(final_ab_carry);

        // ====================================================
        // 2. Prove k*q
        //
        // q is a public fixed constant.
        // ====================================================

        let kq_carry0: AB::Expr =
            row[KQ_CARRY_START].into();

        builder.assert_zero(kq_carry0);

        for out in 0..WIDE_LIMBS {
            let mut acc: AB::Expr =
                row[KQ_CARRY_START + out].into();

            for i in 0..LIMBS {
                if out >= i {
                    let j = out - i;

                    if j < LIMBS {
                        let k_i: AB::Expr =
                            row[K_START + i].into();

                        let q_j =
                            AB::Expr::from_u64(Q[j]);

                        acc = acc + k_i * q_j;
                    }
                }
            }

            let result: AB::Expr =
                row[KQ_START + out].into();

            let next_carry: AB::Expr =
                row[KQ_CARRY_START + out + 1].into();

            builder.assert_zero(
                acc
                    - result
                    - radix.clone() * next_carry
            );
        }

        let final_kq_carry: AB::Expr =
            row[KQ_CARRY_START + WIDE_LIMBS].into();

        builder.assert_zero(final_kq_carry);

        // ====================================================
        // 3. Prove
        //
        // ab = c + kq
        // ====================================================

        let add_carry0: AB::Expr =
            row[ADD_CARRY_START].into();

        builder.assert_zero(add_carry0);

        for i in 0..WIDE_LIMBS {
            let carry_in: AB::Expr =
                row[ADD_CARRY_START + i].into();

            let carry_out: AB::Expr =
                row[ADD_CARRY_START + i + 1].into();

            let kq_i: AB::Expr =
                row[KQ_START + i].into();

            let ab_i: AB::Expr =
                row[AB_START + i].into();

            let c_i =
                if i < LIMBS {
                    let x: AB::Expr =
                        row[C_START + i].into();
                    x
                } else {
                    AB::Expr::ZERO
                };

            builder.assert_zero(
                c_i
                    + kq_i
                    + carry_in
                    - ab_i
                    - radix.clone() * carry_out.clone()
            );

            // Addition carry must be 0 or 1.
            builder.assert_zero(
                carry_out.clone()
                    * (carry_out - AB::Expr::ONE)
            );
        }

        let final_add_carry: AB::Expr =
            row[ADD_CARRY_START + WIDE_LIMBS].into();

        builder.assert_zero(final_add_carry);

        // ====================================================
        // 4. Range checks for all 16-bit limbs
        // ====================================================

        for range_idx in 0..RANGE_LIMBS {
            let value_col =
                if range_idx < LIMBS {
                    A_START + range_idx
                } else if range_idx < 2 * LIMBS {
                    B_START + range_idx - LIMBS
                } else if range_idx < 3 * LIMBS {
                    C_START + range_idx - 2 * LIMBS
                } else if range_idx < 4 * LIMBS {
                    K_START + range_idx - 3 * LIMBS
                } else if range_idx < 4 * LIMBS + WIDE_LIMBS {
                    AB_START
                        + range_idx
                        - 4 * LIMBS
                } else {
                    KQ_START
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
                let bit_col =
                    LIMB_BIT_START
                    + range_idx * LIMB_BITS
                    + bit;

                let b: AB::Expr =
                    row[bit_col].into();

                builder.assert_zero(
                    b.clone()
                        * (b.clone() - AB::Expr::ONE)
                );

                reconstructed =
                    reconstructed
                    + AB::Expr::from_u64(power) * b;

                power <<= 1;
            }

            builder.assert_zero(
                value - reconstructed
            );
        }

        // ====================================================
        // 5. Range-check multiplication carries
        // ====================================================

        for idx in 0..MUL_CARRIES {
            let value_col =
                if idx < 33 {
                    AB_CARRY_START + idx
                } else {
                    KQ_CARRY_START + idx - 33
                };

            let value: AB::Expr =
                row[value_col].into();

            let mut reconstructed =
                AB::Expr::ZERO;

            let mut power = 1u64;

            for bit in 0..CARRY_BITS {
                let bit_col =
                    CARRY_BIT_START
                    + idx * CARRY_BITS
                    + bit;

                let b: AB::Expr =
                    row[bit_col].into();

                builder.assert_zero(
                    b.clone()
                        * (b.clone() - AB::Expr::ONE)
                );

                reconstructed =
                    reconstructed
                    + AB::Expr::from_u64(power) * b;

                power <<= 1;
            }

            builder.assert_zero(
                value - reconstructed
            );
        }
    }
}


// ============================================================
// Helpers
// ============================================================

fn sub_small(
    mut x: [u64; LIMBS],
    amount: u64,
) -> [u64; LIMBS] {
    let mut borrow = amount;

    let mut i = 0;

    while borrow != 0 {
        let low = borrow & 0xffff;

        if x[i] >= low {
            x[i] -= low;
            borrow >>= 16;
        } else {
            x[i] =
                RADIX + x[i] - low;

            borrow =
                (borrow >> 16) + 1;
        }

        i += 1;
    }

    x
}

fn small_limbs(
    value: u64,
) -> [u64; LIMBS] {
    let mut out =
        [0u64; LIMBS];

    out[0] = value;

    out
}

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
                let j = out - i;

                if j < LIMBS {
                    acc +=
                        (a[i] as u128)
                        * (b[j] as u128);
                }
            }
        }

        product[out] =
            (acc % RADIX as u128) as u64;

        carries[out + 1] =
            (acc / RADIX as u128) as u64;
    }

    assert_eq!(
        carries[WIDE_LIMBS],
        0
    );

    (product, carries)
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
            (acc % RADIX as u128)
                as u64;

        carries[i + 1] =
            (acc / RADIX as u128)
                as u64;

        assert_eq!(
            limb,
            expected[i],
            "modular relation failed at limb {}",
            i
        );

        assert!(
            carries[i + 1] <= 1
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
// Trace
// ============================================================

fn make_trace(
    height: usize,
) -> RowMajorMatrix<Val> {
    // a = q - 1
    let a =
        sub_small(Q, 1);

    // b = 2
    let b =
        small_limbs(2);

    // c = q - 2
    let c =
        sub_small(Q, 2);

    // k = 1
    let k =
        small_limbs(1);

    let (ab, ab_carries) =
        multiply_limbs(
            &a,
            &b,
        );

    let (kq, kq_carries) =
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
        Vec::with_capacity(WIDTH);

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

    // ========================================================
    // Range-check bits
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

    for x in ab_carries {
        push_bits(&mut row, x, CARRY_BITS);
    }

    for x in kq_carries {
        push_bits(&mut row, x, CARRY_BITS);
    }

    assert_eq!(
        row.len(),
        WIDTH,
        "bad trace width"
    );

    let mut values =
        Vec::with_capacity(
            WIDTH * height
        );

    for _ in 0..height {
        values.extend_from_slice(&row);
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
        " Plonky3 BN254 Modular Multiplication"
    );
    println!(
        "=============================================="
    );
    println!();

    println!(
        "Native proving field : Goldilocks"
    );
    println!(
        "Target field         : BN254 Fq"
    );
    println!(
        "Relation             : a*b = c + k*q"
    );
    println!();

    println!("Test:");
    println!("  a = q - 1");
    println!("  b = 2");
    println!("  c = q - 2");
    println!("  k = 1");
    println!();

    let trace_height =
        8usize;

    let trace =
        make_trace(trace_height);

    // ========================================================
    // Hash / Merkle
    // ========================================================

    type ByteHash = Sha256;
    type FieldHash =
        SerializingHasher<ByteHash>;

    let byte_hash =
        ByteHash {};

    let field_hash =
        FieldHash::new(byte_hash);

    type Compress =
        CompressionFunctionFromHasher<
            ByteHash,
            2,
            32,
        >;

    let compress =
        Compress::new(byte_hash);

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

    // ========================================================
    // FRI
    // ========================================================

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
            &Bn254ModMulAir,
            trace,
            &[],
        )
        .expect("proving failed");

    let proving_time =
        prove_start.elapsed();

    let proof_bytes =
        postcard::to_allocvec(&proof)
            .expect(
                "serialization failed"
            );

    // ========================================================
    // Verify
    // ========================================================

    let verify_start =
        Instant::now();

    verify(
        &config,
        &Bn254ModMulAir,
        &proof,
        &[],
    )
    .expect(
        "verification failed"
    );

    let verification_time =
        verify_start.elapsed();

    // ========================================================
    // Metrics
    // ========================================================

    let ab_mul_constraints =
        WIDE_LIMBS + 2;

    let kq_mul_constraints =
        WIDE_LIMBS + 2;

    // 32 arithmetic +
    // carry[0] +
    // carry[32] +
    // 32 boolean carry checks
    let addition_constraints =
        WIDE_LIMBS
        + 2
        + WIDE_LIMBS;

    let limb_range_constraints =
        RANGE_LIMBS
        * (LIMB_BITS + 1);

    let carry_range_constraints =
        MUL_CARRIES
        * (CARRY_BITS + 1);

    let total_constraints =
        ab_mul_constraints
        + kq_mul_constraints
        + addition_constraints
        + limb_range_constraints
        + carry_range_constraints;

    println!(
        "=============================================="
    );
    println!(" Metrics");
    println!(
        "=============================================="
    );
    println!();

    println!(
        "a*b constraints            : {}",
        ab_mul_constraints
    );

    println!(
        "k*q constraints            : {}",
        kq_mul_constraints
    );

    println!(
        "Addition constraints       : {}",
        addition_constraints
    );

    println!(
        "Limb range constraints     : {}",
        limb_range_constraints
    );

    println!(
        "Carry range constraints    : {}",
        carry_range_constraints
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
        total_constraints * trace_height
    );

    println!(
        "Proof size                 : {} bytes",
        proof_bytes.len()
    );

    println!(
        "Proving time               : {:?}",
        proving_time
    );

    println!(
        "Verification time          : {:?}",
        verification_time
    );

    println!();

    println!(
        "BN254 modular multiplication verified!"
    );
}