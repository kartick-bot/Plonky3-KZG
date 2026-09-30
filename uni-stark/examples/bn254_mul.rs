use std::time::Instant;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_bn254::Bn254;
use p3_challenger::{HashChallenger, SerializingChallenger32};
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::PrimeCharacteristicRing;
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_sha256::Sha256;
use p3_symmetric::{
    CompressionFunctionFromHasher,
    SerializingHasher,
};
use p3_uni_stark::{prove, verify, StarkConfig};

type Val = Bn254;

// Since BN254 itself is already ~254 bits,
// we use the base field itself as the challenge field here.
type Challenge = Val;


// ============================================================
// AIR
//
// Trace columns:
//
// [ a | b | c ]
//
// Constraint:
//
// c = a * b
// ============================================================

struct MulAir;

impl<F> BaseAir<F> for MulAir {
    fn width(&self) -> usize {
        3
    }
}

impl<AB: AirBuilder> Air<AB> for MulAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.current_slice();

        let a: AB::Expr = row[0].into();
        let b: AB::Expr = row[1].into();
        let c: AB::Expr = row[2].into();

        builder.assert_zero(c - a * b);
    }
}


// ============================================================
// Build trace
// ============================================================

fn make_trace(height: usize) -> RowMajorMatrix<Val> {
    let a = Val::from_u64(12);
    let b = Val::from_u64(17);
    let c = a * b;

    let mut values = Vec::with_capacity(height * 3);

    for _ in 0..height {
        values.push(a);
        values.push(b);
        values.push(c);
    }

    RowMajorMatrix::new(values, 3)
}


// ============================================================
// Main
// ============================================================

fn main() {
    println!("==============================================");
    println!(" Plonky3 BN254 Native Multiplication AIR");
    println!("==============================================");
    println!();

    // --------------------------------------------------------
    // Trace
    // --------------------------------------------------------

    let log_height = 3;
    let trace_height = 1usize << log_height;
    let trace_width = 3;

    let trace = make_trace(trace_height);

    // --------------------------------------------------------
    // Hash / Merkle configuration
    // --------------------------------------------------------

    type ByteHash = Sha256;

    type FieldHash =
        SerializingHasher<ByteHash>;

    let byte_hash = ByteHash {};
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
            0,
        );

    // Challenge == Val in this test.
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

    // --------------------------------------------------------
    // FRI configuration
    // --------------------------------------------------------

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

    // --------------------------------------------------------
    // Fiat-Shamir challenger
    // --------------------------------------------------------

    type Challenger =
        SerializingChallenger32<
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

    // --------------------------------------------------------
    // Prove
    // --------------------------------------------------------

    let prove_start =
        Instant::now();

    let proof =
        prove(
            &config,
            &MulAir,
            trace,
            &[],
        )
        .expect("proof generation failed");

    let prove_time =
        prove_start.elapsed();

    // --------------------------------------------------------
    // Serialize proof
    // --------------------------------------------------------

    let proof_bytes =
        postcard::to_allocvec(&proof)
            .expect(
                "proof serialization failed"
            );

    let proof_size =
        proof_bytes.len();

    // --------------------------------------------------------
    // Verify
    // --------------------------------------------------------

    let verify_start =
        Instant::now();

    verify(
        &config,
        &MulAir,
        &proof,
        &[],
    )
    .expect(
        "verification failed"
    );

    let verify_time =
        verify_start.elapsed();

    // --------------------------------------------------------
    // Metrics
    // --------------------------------------------------------

    let air_constraints = 1;
    let constraint_degree = 2;

    println!("Constraint:");
    println!("    c - a*b = 0");
    println!();

    println!("Example values:");
    println!("    a = 12");
    println!("    b = 17");
    println!("    c = 204");
    println!();

    println!("==============================================");
    println!(" Metrics");
    println!("==============================================");
    println!();

    println!(
        "AIR constraint expressions : {}",
        air_constraints
    );

    println!(
        "Constraint degree          : {}",
        constraint_degree
    );

    println!(
        "Trace width                : {} columns",
        trace_width
    );

    println!(
        "Trace height               : {} rows",
        trace_height
    );

    println!(
        "Trace cells                : {}",
        trace_width * trace_height
    );

    println!(
        "Constraint evaluations     : {}",
        air_constraints * trace_height
    );

    println!(
        "Plonky3 proof size         : {} bytes",
        proof_size
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
        "BN254 multiplication proof verified successfully!"
    );
}