use ark_bn254::{
    Fr,
    G1Affine,
    G1Projective,
    G2Affine,
    G2Projective,
};

use ark_ec::{
    CurveGroup,
    PrimeGroup,
};

use ark_ff::{
    Field,
    PrimeField,
    UniformRand,
    Zero,
};

use rand08::thread_rng;

use sha2::{
    Digest,
    Sha256,
};

use crate::vplustree::VplusTree;


// ---------------------------------------------------------
// Leaf Evaluation Point
// ---------------------------------------------------------

#[derive(Clone, Debug)]
pub struct LeafEvaluation {
    // Actual node index in the V+ Tree.
    pub leaf_node_index: usize,

    // Position of the key inside this leaf.
    pub position_in_leaf: usize,

    // Global index at the leaf level.
    //
    // These are continuous across the entire leaf level:
    //
    // 0, 1, 2, 3, ...
    pub global_index: usize,

    // Original V+ Tree key.
    pub key: i64,

    // SHA256(key || value).
    pub digest: [u8; 32],

    // Polynomial evaluation point.
    //
    // x_i = global_index
    pub x: Fr,

    // Lower 128 bits of the SHA-256 digest,
    // represented exactly as a BN254 field element.
    pub y_low: Fr,

    // Upper 128 bits of the SHA-256 digest,
    // represented exactly as a BN254 field element.
    pub y_high: Fr,
}


// ---------------------------------------------------------
// Leaf Polynomial Input
// ---------------------------------------------------------

#[derive(Clone, Debug)]
pub struct LeafPolynomialData {
    // Actual leaf node index in the V+ Tree.
    pub leaf_node_index: usize,

    // All evaluation points that belong to this leaf.
    pub evaluations: Vec<LeafEvaluation>,
}


// ---------------------------------------------------------
// KZG SRS
// ---------------------------------------------------------

#[derive(Clone, Debug)]
pub struct KzgSrs {
    // [G1, tau*G1, tau^2*G1, ..., tau^d*G1]
    pub g1_powers: Vec<G1Affine>,

    // G2 generator.
    pub g2: G2Affine,

    // tau * G2.
    pub tau_g2: G2Affine,

    // Maximum supported polynomial degree.
    pub max_degree: usize,
}


// ---------------------------------------------------------
// Generate One KZG SRS Per Tree
// ---------------------------------------------------------

pub fn generateTreeSrs(
    max_degree: usize
) -> KzgSrs {

    let mut rng =
        thread_rng();


    // -----------------------------------------------------
    // Sample temporary toxic-waste trapdoor tau.
    //
    // tau exists only inside this function.
    // It is NOT stored in KzgSrs.
    // -----------------------------------------------------

    let tau =
        Fr::rand(
            &mut rng
        );


    // -----------------------------------------------------
    // G1 powers
    //
    // [G1, tau G1, tau^2 G1, ...]
    // -----------------------------------------------------

    let g1_generator =
        G1Projective::generator();


    let mut g1_powers =
        Vec::with_capacity(
            max_degree + 1
        );


    let mut tau_power =
        Fr::from(1u64);


    for _ in 0..=max_degree {

        let point =
            (
                g1_generator
                    * tau_power
            )
            .into_affine();


        g1_powers.push(
            point
        );


        tau_power *=
            tau;
    }


    // -----------------------------------------------------
    // G2 material
    // -----------------------------------------------------

    let g2_generator =
        G2Projective::generator();


    let g2 =
        g2_generator
            .into_affine();


    let tau_g2 =
        (
            g2_generator
                * tau
        )
        .into_affine();


    // -----------------------------------------------------
    // Return only public SRS material.
    //
    // tau is intentionally not returned.
    // -----------------------------------------------------

    KzgSrs {
        g1_powers,
        g2,
        tau_g2,
        max_degree,
    }
}


// ---------------------------------------------------------
// Hash Key-Value Pair
// ---------------------------------------------------------

pub fn hashKeyValue(
    key: i64,
    value: &[u8; 32]
) -> [u8; 32] {

    let mut hasher =
        Sha256::new();


    // Domain separation.
    hasher.update(
        b"VPLUS_LEAF_KEY_VALUE"
    );


    // Encode the key deterministically.
    hasher.update(
        key.to_be_bytes()
    );


    // Include the 32-byte value already stored
    // alongside this key in the leaf.
    hasher.update(
        value
    );


    let result =
        hasher.finalize();


    let mut digest =
        [0u8; 32];


    digest.copy_from_slice(
        &result
    );


    digest
}


// ---------------------------------------------------------
// Split SHA-256 Digest Into Two BN254 Field Elements
// ---------------------------------------------------------

pub fn digestToFieldLimbs(
    digest: &[u8; 32]
) -> (Fr, Fr) {

    // SHA-256 digest:
    //
    // digest[0..16]  = high 128 bits
    // digest[16..32] = low 128 bits
    //
    // Each half is only 128 bits, so it fits
    // exactly inside the BN254 scalar field.

    let high =
        Fr::from_be_bytes_mod_order(
            &digest[0..16]
        );


    let low =
        Fr::from_be_bytes_mod_order(
            &digest[16..32]
        );


    (
        low,
        high
    )
}


// ---------------------------------------------------------
// Build All Leaf-Level Evaluation Points
// ---------------------------------------------------------

pub fn buildLeafEvaluations(
    tree: &VplusTree
) -> Vec<LeafEvaluation> {

    let leaves =
        tree.getAllLeaves();


    let mut evaluations =
        Vec::new();


    // Global index for the entire leaf level.
    //
    // IMPORTANT:
    // This does NOT restart for every leaf.
    let mut global_index =
        0usize;


    for leaf_node_index in leaves {

        let leaf =
            &tree.nodes[leaf_node_index];


        for position_in_leaf in 0..leaf.keys.len() {

            let key =
                leaf.keys[position_in_leaf];


            let value =
                &leaf.values[position_in_leaf];


            // -------------------------------------------------
            // Hash (key, value)
            // -------------------------------------------------

            let digest =
                hashKeyValue(
                    key,
                    value
                );


            // -------------------------------------------------
            // Evaluation point
            //
            // x_i = global index of this key
            // at the leaf level.
            // -------------------------------------------------

            let x =
                Fr::from(
                    global_index as u64
                );


            // -------------------------------------------------
            // Evaluation values
            //
            // Preserve the entire 256-bit SHA-256 digest
            // exactly by splitting it into two 128-bit limbs.
            // -------------------------------------------------

            let (
                y_low,
                y_high
            ) =
                digestToFieldLimbs(
                    &digest
                );


            // -------------------------------------------------
            // Store Evaluation
            // -------------------------------------------------

            evaluations.push(
                LeafEvaluation {
                    leaf_node_index,
                    position_in_leaf,
                    global_index,
                    key,
                    digest,
                    x,
                    y_low,
                    y_high,
                }
            );


            global_index += 1;
        }
    }


    evaluations
}


// ---------------------------------------------------------
// Group Evaluations By Leaf
// ---------------------------------------------------------

pub fn groupEvaluationsByLeaf(
    evaluations: &[LeafEvaluation]
) -> Vec<LeafPolynomialData> {

    let mut grouped:
        Vec<LeafPolynomialData> =
        Vec::new();


    for evaluation in evaluations {

        let mut found =
            false;


        for group in &mut grouped {

            if group.leaf_node_index
                == evaluation.leaf_node_index
            {

                group.evaluations.push(
                    evaluation.clone()
                );

                found =
                    true;

                break;
            }
        }


        if !found {

            grouped.push(
                LeafPolynomialData {
                    leaf_node_index:
                        evaluation.leaf_node_index,

                    evaluations:
                        vec![
                            evaluation.clone()
                        ],
                }
            );
        }
    }


    grouped
}


// ---------------------------------------------------------
// Polynomial Addition
//
// Coefficients are stored in ascending order:
//
// p(X) = c0 + c1 X + c2 X^2 + ...
// ---------------------------------------------------------

fn polynomialAdd(
    a: &[Fr],
    b: &[Fr]
) -> Vec<Fr> {

    let max_len =
        a.len().max(b.len());


    let mut result =
        vec![
            Fr::zero();
            max_len
        ];


    for i in 0..a.len() {

        result[i] +=
            a[i];
    }


    for i in 0..b.len() {

        result[i] +=
            b[i];
    }


    result
}


// ---------------------------------------------------------
// Polynomial Scalar Multiplication
// ---------------------------------------------------------

fn polynomialScale(
    polynomial: &[Fr],
    scalar: Fr
) -> Vec<Fr> {

    polynomial
        .iter()
        .map(
            |coefficient|
                *coefficient * scalar
        )
        .collect()
}


// ---------------------------------------------------------
// Multiply Polynomial By (X - root)
// ---------------------------------------------------------

fn polynomialMultiplyLinear(
    polynomial: &[Fr],
    root: Fr
) -> Vec<Fr> {

    let mut result =
        vec![
            Fr::zero();
            polynomial.len() + 1
        ];


    for i in 0..polynomial.len() {

        // Contribution from -root.
        result[i] -=
            polynomial[i] * root;


        // Contribution from X.
        result[i + 1] +=
            polynomial[i];
    }


    result
}


// ---------------------------------------------------------
// Lagrange Interpolation
//
// Given:
//
// (x_0, y_0), ... , (x_{n-1}, y_{n-1})
//
// returns coefficients of:
//
// f(X) = c0 + c1 X + ... + c_{n-1} X^{n-1}
//
// such that:
//
// f(x_i) = y_i
// ---------------------------------------------------------

pub fn lagrangeInterpolate(
    points: &[(Fr, Fr)]
) -> Vec<Fr> {

    if points.is_empty() {

        return Vec::new();
    }


    let mut result =
        vec![
            Fr::zero()
        ];


    for i in 0..points.len() {

        let (
            x_i,
            y_i
        ) =
            points[i];


        // -------------------------------------------------
        // Build:
        //
        // numerator_i(X)
        // =
        // product_{j != i} (X - x_j)
        // -------------------------------------------------

        let mut basis_polynomial =
            vec![
                Fr::from(1u64)
            ];


        let mut denominator =
            Fr::from(1u64);


        for j in 0..points.len() {

            if i == j {

                continue;
            }


            let (
                x_j,
                _
            ) =
                points[j];


            basis_polynomial =
                polynomialMultiplyLinear(
                    &basis_polynomial,
                    x_j
                );


            denominator *=
                x_i - x_j;
        }


        // -------------------------------------------------
        // Divide by:
        //
        // product_{j != i} (x_i - x_j)
        // -------------------------------------------------

        let denominator_inverse =
            denominator
                .inverse()
                .expect(
                    "Duplicate evaluation points in Lagrange interpolation"
                );


        let scale =
            y_i
                * denominator_inverse;


        let basis_polynomial =
            polynomialScale(
                &basis_polynomial,
                scale
            );


        result =
            polynomialAdd(
                &result,
                &basis_polynomial
            );
    }


    result
}


// ---------------------------------------------------------
// Leaf Polynomials
// ---------------------------------------------------------

#[derive(Clone, Debug)]
pub struct LeafPolynomials {
    pub leaf_node_index: usize,

    // Lagrange polynomial for low 128-bit limbs.
    pub low_coefficients: Vec<Fr>,

    // Lagrange polynomial for high 128-bit limbs.
    pub high_coefficients: Vec<Fr>,

    // KZG commitment to the low polynomial.
    pub low_commitment: G1Affine,

    // KZG commitment to the high polynomial.
    pub high_commitment: G1Affine,
}


// ---------------------------------------------------------
// Commit To Polynomial Using Public KZG SRS
// ---------------------------------------------------------

pub fn commitPolynomial(
    coefficients: &[Fr],
    srs: &KzgSrs
) -> G1Affine {

    assert!(
        coefficients.len()
            <= srs.g1_powers.len(),
        "Polynomial degree exceeds SRS capacity"
    );


    let mut commitment =
        G1Projective::zero();


    for (
        coefficient,
        srs_power
    ) in coefficients
        .iter()
        .zip(
            srs.g1_powers.iter()
        )
    {

        commitment +=
            G1Projective::from(
                *srs_power
            )
            * coefficient;
    }


    commitment
        .into_affine()
}


// ---------------------------------------------------------
// Build One Pair Of Polynomials Per Leaf
// ---------------------------------------------------------

pub fn buildLeafPolynomials(
    grouped: &[LeafPolynomialData],
    srs: &KzgSrs
) -> Vec<LeafPolynomials> {

    let mut result =
        Vec::new();


    for leaf_group in grouped {

        let mut low_points =
            Vec::new();


        let mut high_points =
            Vec::new();


        for evaluation in
            &leaf_group.evaluations
        {

            low_points.push(
                (
                    evaluation.x,
                    evaluation.y_low
                )
            );


            high_points.push(
                (
                    evaluation.x,
                    evaluation.y_high
                )
            );
        }


        // -------------------------------------------------
        // Lagrange interpolation
        // -------------------------------------------------

        let low_coefficients =
            lagrangeInterpolate(
                &low_points
            );


        let high_coefficients =
            lagrangeInterpolate(
                &high_points
            );


        // -------------------------------------------------
        // KZG commitments
        // -------------------------------------------------

        let low_commitment =
            commitPolynomial(
                &low_coefficients,
                srs
            );


        let high_commitment =
            commitPolynomial(
                &high_coefficients,
                srs
            );


        result.push(
            LeafPolynomials {
                leaf_node_index:
                    leaf_group.leaf_node_index,

                low_coefficients,

                high_coefficients,

                low_commitment,

                high_commitment,
            }
        );
    }


    result
}


// ---------------------------------------------------------
// Evaluate Polynomial
// ---------------------------------------------------------

pub fn evaluatePolynomial(
    coefficients: &[Fr],
    x: Fr
) -> Fr {

    // Horner evaluation.
    let mut result =
        Fr::from(
            0u64
        );


    for coefficient
        in coefficients.iter().rev()
    {

        result *=
            x;


        result +=
            coefficient;
    }


    result
}


// ---------------------------------------------------------
// Print Leaf Polynomials And Verify Interpolation
// ---------------------------------------------------------

pub fn printAndVerifyLeafPolynomials(
    grouped: &[LeafPolynomialData],
    polynomials: &[LeafPolynomials]
) {

    println!();

    println!(
        "=========================================="
    );

    println!(
        "Leaf-Level Lagrange Polynomials"
    );

    println!(
        "=========================================="
    );


    for polynomial in polynomials {

        println!();


        println!(
            "Leaf Node Index: {}",
            polynomial.leaf_node_index
        );


        // -------------------------------------------------
        // Low polynomial
        // -------------------------------------------------

        println!(
            "Low Polynomial Coefficients:"
        );


        for (
            degree,
            coefficient
        ) in polynomial
            .low_coefficients
            .iter()
            .enumerate()
        {

            println!(
                "  X^{} -> {}",
                degree,
                coefficient
            );
        }


        // -------------------------------------------------
        // High polynomial
        // -------------------------------------------------

        println!(
            "High Polynomial Coefficients:"
        );


        for (
            degree,
            coefficient
        ) in polynomial
            .high_coefficients
            .iter()
            .enumerate()
        {

            println!(
                "  X^{} -> {}",
                degree,
                coefficient
            );
        }


        // -------------------------------------------------
        // KZG commitments
        // -------------------------------------------------

        println!(
            "Low KZG Commitment: {:?}",
            polynomial.low_commitment
        );


        println!(
            "High KZG Commitment: {:?}",
            polynomial.high_commitment
        );


        // -------------------------------------------------
        // Find corresponding leaf evaluations
        // -------------------------------------------------

        let leaf_group =
            grouped
                .iter()
                .find(
                    |group|
                        group.leaf_node_index
                            == polynomial.leaf_node_index
                )
                .expect(
                    "Leaf group not found"
                );


        // -------------------------------------------------
        // Verify interpolation
        // -------------------------------------------------

        println!(
            "Interpolation Verification:"
        );


        for evaluation in
            &leaf_group.evaluations
        {

            let low_result =
                evaluatePolynomial(
                    &polynomial.low_coefficients,
                    evaluation.x
                );


            let high_result =
                evaluatePolynomial(
                    &polynomial.high_coefficients,
                    evaluation.x
                );


            println!(
                "  index={} key={} low_match={} high_match={}",
                evaluation.global_index,
                evaluation.key,
                low_result
                    == evaluation.y_low,
                high_result
                    == evaluation.y_high
            );
        }
    }
}