use ark_bn254::{Bn254, Fr, G1Projective, G2Projective};
use ark_ec::{pairing::Pairing, CurveGroup, PrimeGroup};
use ark_serialize::CanonicalSerialize;
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
use p3_uni_stark::{prove, verify, StarkConfig};

fn main() {
    // Step 1: Define polynomial coefficients
    // f(X) = 7 + 5X + 3X^2
    let coefficients = [Fr::from(7u64), Fr::from(5u64), Fr::from(3u64)];

    // Step 2: Toy setup parameter
    let tau = Fr::from(11u64);

    // Step 3: Generate SRS in G1
    let g1 = G1Projective::generator();

    let srs = [g1, g1 * tau, g1 * tau * tau];

    // Step 4: Compute KZG commitment
    let mut commitment = G1Projective::default();

    for i in 0..coefficients.len() {
        commitment += srs[i] * coefficients[i];
    }

    // Step 5: Calculate f(tau) for comparison
    let f_tau = coefficients[0] + coefficients[1] * tau + coefficients[2] * tau * tau;

    let expected_commitment = g1 * f_tau;

    assert_eq!(commitment, expected_commitment);

    println!("Polynomial: f(X) = 3X^2 + 5X + 7");
    println!("Toy setup tau = 11");
    println!("f(tau) = 425");

    println!("KZG commitment: {:?}", commitment.into_affine());

    println!("KZG commitment constructed successfully!");

    // Step 3: Generate a KZG opening proof.

    // Evaluation point z = 2
    let z = Fr::from(2u64);

    // Claimed evaluation f(2) = 29
    let y = Fr::from(29u64);

    // Quotient polynomial:
    // q(X) = (f(X) - 29) / (X - 2)
    //      = 3X + 11

    let quotient_coefficients = [Fr::from(11u64), Fr::from(3u64)];

    // Commit to the quotient polynomial
    // using the same SRS.
    let mut opening_proof = G1Projective::default();

    for i in 0..quotient_coefficients.len() {
        opening_proof += srs[i] * quotient_coefficients[i];
    }

    // Check q(tau) = 3*11 + 11 = 44
    let q_tau = quotient_coefficients[0] + quotient_coefficients[1] * tau;

    assert_eq!(q_tau, Fr::from(44u64));
    assert_eq!(opening_proof, g1 * q_tau);

    // Check polynomial identity at tau:
    // f(tau) - y = q(tau) * (tau - z)
    assert_eq!(f_tau - y, q_tau * (tau - z));

    println!("Evaluation point z = 2");
    println!("Claimed evaluation y = 29");
    println!("Quotient polynomial: q(X) = 3X + 11");

    println!("KZG opening proof: {:?}", opening_proof.into_affine());

    println!("KZG opening proof generated successfully!");

    // Step 4: Verify the KZG opening using pairings.

    // Generate G2 and the public SRS element [tau]G2.
    let g2 = G2Projective::generator();
    let tau_g2 = g2 * tau;

    // Left pairing input:
    // C_f - [y]G1
    let lhs_g1 = commitment - g1 * y;

    // Right pairing input:
    // [tau]G2 - [z]G2
    let rhs_g2 = tau_g2 - g2 * z;

    // Compute both pairings.
    let lhs = Bn254::pairing(lhs_g1.into_affine(), g2.into_affine());

    let rhs = Bn254::pairing(opening_proof.into_affine(), rhs_g2.into_affine());

    // Verify the pairing equation.
    assert_eq!(lhs, rhs, "KZG pairing verification failed!");

    println!("KZG pairing equation satisfied!");
    println!("KZG opening verified successfully!");

    // Step 5: Represent KZG points in BabyBear.

    // Serialize commitment C_f.
    let mut commitment_bytes = Vec::new();

    commitment
        .into_affine()
        .serialize_compressed(&mut commitment_bytes)
        .unwrap();

    // Serialize opening proof pi.
    let mut proof_bytes = Vec::new();

    opening_proof
        .into_affine()
        .serialize_compressed(&mut proof_bytes)
        .unwrap();

    println!("Commitment bytes: {}", commitment_bytes.len());

    println!("Opening proof bytes: {}", proof_bytes.len());

    // Each byte fits into a BabyBear element.
    assert_eq!(commitment_bytes.len(), proof_bytes.len());

    let mut values = Vec::new();

    for i in 0..commitment_bytes.len() {
        let c = BabyBear::from_u32(commitment_bytes[i] as u32);
        let p = BabyBear::from_u32(proof_bytes[i] as u32);
        let sum = c + p;

        values.push(c);
        values.push(p);
        values.push(sum);
    }

    let trace = RowMajorMatrix::new(values, 3);

    println!("\nPlonky3 KZG execution trace:");

    for (i, row) in trace.values.chunks(3).enumerate() {
        println!(
            "Row {}: C = {}, pi = {}, sum = {}",
            i,
            row[0].as_canonical_u32(),
            row[1].as_canonical_u32(),
            row[2].as_canonical_u32()
        );
    }
    struct KzgAir;

    impl<F> BaseAir<F> for KzgAir {
        fn width(&self) -> usize {
            3
        }

        fn main_next_row_columns(&self) -> Vec<usize> {
            vec![]
        }
    }

    impl<AB: AirBuilder> Air<AB> for KzgAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.current_slice();

            let c = row[0].clone();
            let p = row[1].clone();
            let sum = row[2].clone();

            builder.assert_zero(c + p - sum);
        }
    }

    let air = KzgAir;

    println!("\nAIR constructed successfully!");

    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;

    type ByteHash = Sha256;
    type FieldHash = SerializingHasher<ByteHash>;

    let byte_hash = ByteHash {};
    let field_hash = FieldHash::new(Sha256);

    type MyCompress = CompressionFunctionFromHasher<ByteHash, 2, 32>;

    let compress = MyCompress::new(byte_hash);

    type ValMmcs = MerkleTreeMmcs<Val, u8, FieldHash, MyCompress, 2, 32>;

    let val_mmcs = ValMmcs::new(field_hash, compress, 3);

    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;

    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    type Dft = p3_dft::Radix2Bowers;

    let dft = Dft::default();

    type Challenger = SerializingChallenger32<Val, HashChallenger<u8, ByteHash, 32>>;

    let challenger = Challenger::from_hasher(vec![], byte_hash);

    let fri_params = FriParameters::new_benchmark(challenge_mmcs);

    type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;

    let pcs = Pcs::new(dft, val_mmcs, fri_params);

    type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;

    let config = MyConfig::new(pcs, challenger);

    println!("\nGenerating STARK proof...");

    let proof = prove(&config, &air, trace, &[]).unwrap();

    println!("STARK proof generated successfully!");

    println!("Verifying STARK proof...");

    verify(&config, &air, &proof, &[]).expect("STARK verification failed");

    println!("STARK proof verified successfully!");
}
