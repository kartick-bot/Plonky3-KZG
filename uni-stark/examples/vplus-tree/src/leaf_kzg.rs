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
use ark_serialize::CanonicalSerialize;



// ---------------------------------------------------------
// Leaf Evaluation Record
// ---------------------------------------------------------

#[derive(Clone, Debug)]
pub struct LeafEvaluation {

    // Actual node index in the V+ Tree.
    pub leaf_node_index: usize,

    // Position of the key inside this leaf.
    pub position_in_leaf: usize,

    // Logical global index at the leaf level.
    //
    // These are continuous across the entire leaf level:
    //
    // 0, 1, 2, 3, ...
    pub global_index: usize,

    // Original V+ Tree key.
    pub key: i64,

    // SHA256(key || value).
    pub digest: [u8; 32],

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

    // All logical records that belong to this leaf.
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
        Fr::from(
            1u64
        );


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
// Build All Leaf-Level Logical Evaluations
// ---------------------------------------------------------

pub fn buildLeafEvaluations(
    tree: &VplusTree
) -> Vec<LeafEvaluation> {

    let leaves =
        tree.getAllLeaves();


    let mut evaluations =
        Vec::new();


    // Logical index for the entire leaf level.
    //
    // IMPORTANT:
    //
    // This does NOT restart for every leaf.
    //
    // Every (key,value) record gets one logical index:
    //
    // 0, 1, 2, 3, ...
    let mut global_index =
        0usize;


    for leaf_node_index in leaves {

        let leaf =
            &tree.nodes[
                leaf_node_index
            ];


        for position_in_leaf
            in 0..leaf.keys.len()
        {

            let key =
                leaf.keys[
                    position_in_leaf
                ];


            let value =
                &leaf.values[
                    position_in_leaf
                ];


            // -------------------------------------------------
            // Hash (key, value)
            // -------------------------------------------------

            let digest =
                hashKeyValue(
                    key,
                    value
                );


            // -------------------------------------------------
            // Preserve the complete 256-bit digest.
            //
            // We split it into two exact 128-bit BN254
            // field elements.
            // -------------------------------------------------

            let (
                y_low,
                y_high
            ) =
                digestToFieldLimbs(
                    &digest
                );


            // -------------------------------------------------
            // Store Logical Record
            // -------------------------------------------------

            evaluations.push(
                LeafEvaluation {

                    leaf_node_index,

                    position_in_leaf,

                    global_index,

                    key,

                    digest,

                    y_low,

                    y_high,
                }
            );


            global_index +=
                1;
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
        a.len()
            .max(
                b.len()
            );


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
                *coefficient
                    * scalar
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
            polynomial[i]
                * root;


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
                Fr::from(
                    1u64
                )
            ];


        let mut denominator =
            Fr::from(
                1u64
            );


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
// Leaf Polynomial
//
// IMPORTANT:
//
// We create ONE polynomial per leaf.
//
// Every logical (key,value) record contributes one
// interpolation point. The two field limbs are combined as:
//
//     y = y_low + y_high
//
// For position i inside the leaf:
//
//     f(i) = y
//
// With at most (order - 1) keys in a leaf, the maximum
// polynomial degree is (order - 2).
// ---------------------------------------------------------

#[derive(Clone, Debug)]
pub struct LeafPolynomial {

    pub leaf_node_index: usize,

    // One polynomial containing both SHA-256 limbs.
    pub coefficients: Vec<Fr>,

    // One KZG commitment to the polynomial.
    pub commitment: G1Affine,
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
// Build One Polynomial Per Leaf
// ---------------------------------------------------------

pub fn buildLeafPolynomials(
    grouped: &[LeafPolynomialData],
    srs: &KzgSrs
) -> Vec<LeafPolynomial> {

    let mut result =
        Vec::new();


    for leaf_group in grouped {

        // -------------------------------------------------
        // ONE point set for the leaf.
        //
        // If logical record i contains:
        //
        // (y_low, y_high)
        //
        // then:
        //
        // f(i) = y_low + y_high
        // -------------------------------------------------

        let mut points =
            Vec::new();


for evaluation in
    &leaf_group.evaluations
{
    let x =
        Fr::from(
            evaluation.position_in_leaf as u64
        );


    let y =
        evaluation.y_low
        +
        evaluation.y_high;


    points.push(
        (
            x,
            y
        )
    );
}


        // -------------------------------------------------
        // Lagrange interpolation
        // -------------------------------------------------

        let coefficients =
            lagrangeInterpolate(
                &points
            );


        // -------------------------------------------------
        // ONE KZG commitment
        // -------------------------------------------------

        let commitment =
            commitPolynomial(
                &coefficients,
                srs
            );


        result.push(
            LeafPolynomial {

                leaf_node_index:
                    leaf_group.leaf_node_index,

                coefficients,

                commitment,
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
    polynomials: &[LeafPolynomial]
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
        // Polynomial coefficients
        // -------------------------------------------------

        println!(
            "Polynomial Coefficients:"
        );


        for (
            degree,
            coefficient
        ) in polynomial
            .coefficients
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
        // ONE KZG commitment
        // -------------------------------------------------

        println!(
            "KZG Commitment: {:?}",
            polynomial.commitment
        );


        // -------------------------------------------------
        // Find corresponding logical leaf evaluations
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

            let position =
                evaluation.position_in_leaf;


            let x =
                Fr::from(
                    position as u64
                );


            let expected =
                evaluation.y_low
                +
                evaluation.y_high;


            let result =
                evaluatePolynomial(
                    &polynomial.coefficients,
                    x
                );


            println!(
                "  position={} key={} match={}",
                position,
                evaluation.key,
                result == expected
            );
        }
    }
}

 // Parent commitment

fn commitmentToBytes(
    commitment: &G1Affine
) -> Vec<u8> {

    let mut bytes =
        Vec::new();

    commitment
        .serialize_compressed(
            &mut bytes
        )
        .expect(
            "Could not serialize commitment"
        );

    bytes
}


// Hash key + left commitment + right commitment

fn hashInternalRecord(
    key: i64,
    leftCommitment: &G1Affine,
    rightCommitment: &G1Affine
) -> [u8; 32] {

    let mut hasher =
        Sha256::new();


    hasher.update(
        b"VPLUS_INTERNAL_NODE"
    );


    hasher.update(
        key.to_be_bytes()
    );


    let left_bytes =
        commitmentToBytes(
            leftCommitment
        );


    let right_bytes =
        commitmentToBytes(
            rightCommitment
        );


    hasher.update(
        &left_bytes
    );


    hasher.update(
        &right_bytes
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


// Build commitment for one node

fn buildNodeCommitment(
    tree: &VplusTree,
    node_index: usize,
    leaf_polynomials: &[LeafPolynomial],
    srs: &KzgSrs
) -> G1Affine {

    let node =
        &tree.nodes[
            node_index
        ];


    // Leaf: return existing commitment

    if node.is_leaf {

        let leaf =
            leaf_polynomials
                .iter()
                .find(
                    |leaf|
                        leaf.leaf_node_index
                            == node_index
                )
                .expect(
                    "Leaf commitment not found"
                );


        return leaf.commitment;
    }


    // Get child commitments

    let mut child_commitments =
        Vec::new();


    for child_index in
        &node.children
    {

        let commitment =
            buildNodeCommitment(
                tree,
                *child_index,
                leaf_polynomials,
                srs
            );


        child_commitments.push(
            commitment
        );
    }


    // Build parent points

    let mut points =
        Vec::new();


    for i in 0..node.keys.len() {

        let key =
            node.keys[i];


        let left_commitment =
            &child_commitments[i];


        let right_commitment =
            &child_commitments[i + 1];


        let digest =
            hashInternalRecord(
                key,
                left_commitment,
                right_commitment
            );


        let (
            y_low,
            y_high
        ) =
            digestToFieldLimbs(
                &digest
            );


        let x =
            Fr::from(
                i as u64
            );


        let y =
            y_low
            +
            y_high;


        points.push(
            (
                x,
                y
            )
        );
    }


    // Build polynomial

    let coefficients =
        lagrangeInterpolate(
            &points
        );


    // Build commitment

    let commitment =
        commitPolynomial(
            &coefficients,
            srs
        );


    println!(
        "Internal Node {} Commitment: {:?}",
        node_index,
        commitment
    );


    commitment
}


// Build all commitments up to root

pub fn buildTreeCommitment(
    tree: &VplusTree,
    leaf_polynomials: &[LeafPolynomial],
    srs: &KzgSrs
) -> G1Affine {

    let root_commitment =
        buildNodeCommitment(
            tree,
            tree.root,
            leaf_polynomials,
            srs
        );


    println!(
        "Root Commitment: {:?}",
        root_commitment
    );


    root_commitment
}                  

// ---------------------------------------------------------
// Compute KZG Witness For One Polynomial
// ---------------------------------------------------------

pub fn computeKzgWitness(
    coefficients: &[Fr],
    i: Fr,
    srs: &KzgSrs
) -> G1Affine {

    if coefficients.is_empty() {

        panic!(
            "Cannot compute witness for empty polynomial"
        );
    }


    // f(i)

    let y =
        evaluatePolynomial(
            coefficients,
            i
        );


    // f(X) - f(i)

    let mut numerator =
        coefficients
            .to_vec();


    numerator[0] -=
        y;


    // f(i) - f(i) must be zero

    assert!(
        evaluatePolynomial(
            &numerator,
            i
        )
        == Fr::zero(),
        "Witness error: numerator is not zero at i"
    );


    // Constant polynomial

    if coefficients.len()
        == 1
    {

        let quotient =
            vec![
                Fr::zero()
            ];


        return commitPolynomial(
            &quotient,
            srs
        );
    }


    // -----------------------------------------------------
    // Synthetic Division
    //
    // q(X) =
    //
    // f(X) - f(i)
    // -------------
    //    X - i
    // -----------------------------------------------------

    let mut quotient =
        vec![
            Fr::zero();
            numerator.len() - 1
        ];


    let highest =
        quotient.len() - 1;


    quotient[highest] =
        numerator[
            numerator.len() - 1
        ];


    if quotient.len()
        > 1
    {

        for j in
            (
                1..
                quotient.len()
            )
            .rev()
        {

            quotient[j - 1] =
                numerator[j]
                +
                i
                *
                quotient[j];
        }
    }


    // -----------------------------------------------------
    // Division Remainder
    //
    // numerator =
    //
    // (X - i)q(X) + remainder
    //
    // therefore:
    //
    // remainder =
    //
    // numerator[0] + i * quotient[0]
    // -----------------------------------------------------

    let remainder =
        numerator[0]
        +
        i
        *
        quotient[0];


    assert!(
        remainder
            == Fr::zero(),
        "Witness error: polynomial division remainder is not zero"
    );


    // -----------------------------------------------------
    // Reconstruct:
    //
    // q(X)(X - i)
    //
    // and verify that it equals:
    //
    // f(X) - f(i)
    // -----------------------------------------------------

    let mut reconstructed =
        vec![
            Fr::zero();
            numerator.len()
        ];


    for j in
        0..quotient.len()
    {

        reconstructed[j] -=
            quotient[j]
            *
            i;


        reconstructed[j + 1] +=
            quotient[j];
    }


    assert_eq!(
        reconstructed,
        numerator,
        "Witness error: polynomial reconstruction failed"
    );


    // -----------------------------------------------------
    // KZG Witness
    //
    // witness = Commit(q)
    // -----------------------------------------------------

    let witness =
        commitPolynomial(
            &quotient,
            srs
        );


    witness
}


// ---------------------------------------------------------
// Compute KZG Witness Recursively
//
// Starts from the root.
//
// Recursion follows:
//
// Root
//   -> Internal Node
//       -> Internal Node
//           -> Leaf
//
// Witnesses are added while recursion returns:
//
// Leaf
//   -> Parent
//       -> Parent
//           -> Root
// ---------------------------------------------------------

pub fn computeKzgWitnessRecursive(
    tree: &VplusTree,
    node_index: usize,
    searchKey: i64,
    leaf_polynomials: &[LeafPolynomial],
    srs: &KzgSrs
) -> Vec<G1Affine> {

    // -----------------------------------------------------
    // Leaf
    // -----------------------------------------------------

    if tree.nodes[node_index]
        .is_leaf
    {

        let leaf =
            leaf_polynomials
                .iter()
                .find(
                    |leaf|
                        leaf.leaf_node_index
                            == node_index
                )
                .expect(
                    "Leaf polynomial not found"
                );


        let mut i =
            0usize;


        while i
            < tree.nodes[node_index]
                .keys
                .len()
            &&
            tree.nodes[node_index]
                .keys[i]
                != searchKey
        {

            i +=
                1;
        }


        if i
            == tree.nodes[node_index]
                .keys
                .len()
        {

            panic!(
                "Search key not found in leaf"
            );
        }


        // Verify the polynomial commitment before witness

        assert_eq!(
            commitPolynomial(
                &leaf.coefficients,
                srs
            ),
            leaf.commitment,
            "Leaf commitment does not match leaf polynomial"
        );


        let witness =
            computeKzgWitness(
                &leaf.coefficients,
                Fr::from(
                    i as u64
                ),
                srs
            );


        return vec![
            witness
        ];
    }


    // -----------------------------------------------------
    // Find Child Position
    //
    // Same search rule already used by V+ Tree
    // -----------------------------------------------------

    let mut childPosition =
        0usize;


    while childPosition
        < tree.nodes[node_index]
            .keys
            .len()
        &&
        searchKey
            >= tree.nodes[node_index]
                .keys[
                    childPosition
                ]
    {

        childPosition +=
            1;
    }


    let child_index =
        tree.nodes[node_index]
            .children[
                childPosition
            ];


    // -----------------------------------------------------
    // Recursive Call
    // -----------------------------------------------------

    let mut witnesses =
        computeKzgWitnessRecursive(
            tree,
            child_index,
            searchKey,
            leaf_polynomials,
            srs
        );


    // -----------------------------------------------------
    // Build Commitments Of Children
    // -----------------------------------------------------

    let mut child_commitments =
        Vec::new();


    for child_index in
        &tree.nodes[node_index]
            .children
    {

        let commitment =
            buildNodeCommitment(
                tree,
                *child_index,
                leaf_polynomials,
                srs
            );


        child_commitments.push(
            commitment
        );
    }


    // -----------------------------------------------------
    // Build Same Internal Polynomial
    // Used By buildNodeCommitment()
    // -----------------------------------------------------

    let mut points =
        Vec::new();


    for j in
        0..tree.nodes[node_index]
            .keys
            .len()
    {

        let key =
            tree.nodes[node_index]
                .keys[j];


        let digest =
            hashInternalRecord(
                key,
                &child_commitments[j],
                &child_commitments[j + 1]
            );


        let (
            y_low,
            y_high
        ) =
            digestToFieldLimbs(
                &digest
            );


        let i =
            Fr::from(
                j as u64
            );


        let y =
            y_low
            +
            y_high;


        points.push(
            (
                i,
                y
            )
        );
    }


    let coefficients =
        lagrangeInterpolate(
            &points
        );


    // -----------------------------------------------------
    // Verify Internal Polynomial Commitment
    // -----------------------------------------------------

    assert_eq!(
        commitPolynomial(
            &coefficients,
            srs
        ),
        buildNodeCommitment(
            tree,
            node_index,
            leaf_polynomials,
            srs
        ),
        "Internal node commitment does not match polynomial"
    );


    // -----------------------------------------------------
    // Select Record Containing Chosen Child Commitment
    //
    // C0 -> record 0 LPC
    // C1 -> record 1 LPC
    // C2 -> record 2 LPC
    //
    // Last child:
    //
    // Cn -> last record RPC
    // -----------------------------------------------------

    let i =
        if childPosition
            < tree.nodes[node_index]
                .keys
                .len()
        {

            childPosition
        }
        else {

            tree.nodes[node_index]
                .keys
                .len()
                - 1
        };


    // -----------------------------------------------------
    // Compute Witness For This Internal Node
    // -----------------------------------------------------

    let witness =
        computeKzgWitness(
            &coefficients,
            Fr::from(
                i as u64
            ),
            srs
        );


    witnesses.push(
        witness
    );


    witnesses
}