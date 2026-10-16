use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_baby_bear::BabyBear;
use p3_challenger::{HashChallenger, SerializingChallenger32};
use p3_commit::ExtensionMmcs;
use p3_field::extension::BinomialExtensionField;
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_sha256::Sha256;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use p3_uni_stark::{StarkConfig, prove, verify};

// Step 1: Define the AIR structure.
struct RangeCheckAir;

// Step 2: Define the trace width.
impl<F> BaseAir<F> for RangeCheckAir {
    fn width(&self) -> usize {
        2
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        vec![]
    }
}

// Step 3: Define the AIR constraint.
impl<AB: AirBuilder> Air<AB> for RangeCheckAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.current_slice();

        let x = row[0].clone();
        let y = row[1].clone();

        // Constraint 1: x + y = 10
        builder.assert_zero(x.clone() + y - AB::Expr::from_u32(10));

        // Constraint 2: x belongs to {0, 1, ..., 9}
        let mut product = AB::Expr::ONE;

        for i in 0..10 {
            product = product * (x.clone() - AB::Expr::from_u32(i));
        }

        builder.assert_zero(product);
    }
}

// Step 4: Main function.
fn main() {
    // Sample x between 0 and 9.
    let x: u32 = rand::random_range(0..10);

    // Convert to BabyBear field elements.
    let x_field = BabyBear::from_u32(x);
    let ten = BabyBear::from_u32(10);
    let y_field = ten - x_field;

    println!("x in field = {}", x_field.as_canonical_u32());
    println!("y in field = {}", y_field.as_canonical_u32());

    // Construct the execution trace.
    let mut values = Vec::new();

    const TRACE_LENGTH: usize = 32;

    for _ in 0..TRACE_LENGTH {
        values.push(x_field);
        values.push(y_field);
    }

    let trace = RowMajorMatrix::new(values, 2);

    println!("Sampled x = {}", x);
    println!("Execution Trace:");

    for row in trace.values.chunks(2) {
        println!(
            "x = {}, y = {}",
            row[0].as_canonical_u32(),
            row[1].as_canonical_u32()
        );
    }

    // Construct our AIR.
    let _air = RangeCheckAir;
    println!("\nChecking trace constraints:");
    println!("\nChecking AIR constraints using Plonky3...");

    p3_air::check_constraints(&_air, &trace, &[]);

    println!("Plonky3 AIR constraints satisfied!");
    // Step 15: Configure the STARK prover.

    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;

    // SHA-256 hashing
    type ByteHash = Sha256;
    type FieldHash = SerializingHasher<ByteHash>;

    let byte_hash = ByteHash {};
    let field_hash = FieldHash::new(Sha256);

    // Merkle tree compression
    type MyCompress = CompressionFunctionFromHasher<ByteHash, 2, 32>;

    let compress = MyCompress::new(byte_hash);

    // Commitments over BabyBear
    type ValMmcs = MerkleTreeMmcs<Val, u8, FieldHash, MyCompress, 2, 32>;

    let val_mmcs = ValMmcs::new(field_hash, compress, 3);

    // Commitments over the extension field
    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;

    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    // Discrete Fourier Transform
    type Dft = p3_dft::Radix2Bowers;
    let dft = Dft::default();

    // Fiat-Shamir challenger
    type Challenger = SerializingChallenger32<Val, HashChallenger<u8, ByteHash, 32>>;

    let challenger = Challenger::from_hasher(vec![], byte_hash);

    // FRI parameters
    let fri_params = FriParameters::new_benchmark(challenge_mmcs);

    // Polynomial commitment scheme
    type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;

    let pcs = Pcs::new(dft, val_mmcs, fri_params);

    // Final STARK configuration
    type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;

    let config = MyConfig::new(pcs, challenger);

    println!("Generating STARK proof...");

    let proof =
        prove(&config, &RangeCheckAir, trace.clone(), &[]).expect("STARK proof generation failed");

    println!("STARK proof generated successfully!");

    println!("Verifying STARK proof...");

    verify(&config, &RangeCheckAir, &proof, &[]).expect("STARK proof verification failed");

    println!("STARK proof verified successfully!");

    println!("STARK configuration initialized!");
    for (i, row) in trace.values.chunks(2).enumerate() {
        let x = row[0];
        let y = row[1];

        // Evaluate x + y - 10
        let constraint = x + y - ten;

        println!("Row {}: x + y - 10 = {}", i, constraint.as_canonical_u32());

        assert_eq!(constraint, BabyBear::ZERO, "Constraint failed at row {}", i);
    }

    println!("All trace constraints satisfied!");

    println!("AIR constructed successfully!");
}
