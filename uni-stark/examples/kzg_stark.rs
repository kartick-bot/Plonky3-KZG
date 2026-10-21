//! JSON -> real V+ Tree membership witness -> BabyBear encoding.
//!
//! This module deliberately keeps the prover/private data separate from the
//! verifier/public data.
//!
//! It also re-runs the existing Arkworks KZG verification for every opening
//! from the selected leaf to the trusted root before any STARK trace is built.

use ark_bn254::{Fq, Fq2, Fr, G1Affine, G2Affine};
use ark_ff::PrimeField;
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
use p3_sha256::Sha256 as P3Sha256;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use p3_uni_stark::{prove, verify, StarkConfig};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::{Duration, Instant};

use crate::leaf_kzg::{
    commitPolynomial,
    digestToFieldLimbs,
    evaluatePolynomial,
    hashKeyValue,
    internalRecordEvaluation,
    lagrangeInterpolate,
    KzgSrs,
};
use crate::verify_eval::verifyEval;


// ---------------------------------------------------------
// Existing Single KZG Opening Interface
// ---------------------------------------------------------

#[derive(Clone, Debug)]
pub struct VplusKzgWitness {
    pub commitment: G1Affine,
    pub evaluation_point: Fr,
    pub evaluation_value: Fr,
    pub opening_witness: G1Affine,
    pub g1: G1Affine,
    pub g2: G2Affine,
    pub tau_g2: G2Affine,
}


// ---------------------------------------------------------
// Public Query Path Step
//
// Verifier receives ONLY the path data for the selected queried key.
//
// IMPORTANT:
//
// This path is the selected leaf -> root path only. It is NOT
// data for the entire tree. Internal-node polynomial coefficients
// and internal-node KZG opening witnesses remain prover-private.
// ---------------------------------------------------------

#[derive(Clone, Debug)]
pub struct VplusPublicPathStep {
    pub node_index: usize,
    pub record_position: usize,
    pub mapping: String,
    pub separator_key: i64,
    pub left_commitment: G1Affine,
    pub right_commitment: G1Affine,
    pub evaluation_point: Fr,
    pub evaluation_value: Fr,
    pub parent_commitment: G1Affine,
}


// ---------------------------------------------------------
// Public Statement
//
// Verifier receives:
//
//     selected key
//     trusted root commitment
//     selected leaf commitment
//     selected leaf evaluation point
//     selected leaf evaluation value
//     selected leaf polynomial coefficients
//     selected leaf KZG opening witness
//     selected leaf -> root path data only
//
// The verifier does NOT receive coefficients or KZG opening
// witnesses for internal path nodes, unrelated leaves, or unrelated
// tree nodes. It also does not receive y_low, y_high, mapped values,
// or the private STARK trace.
// ---------------------------------------------------------

#[derive(Clone, Debug)]
pub struct VplusPublicStatement {
    pub key: i64,
    pub root_commitment: G1Affine,
    pub leaf_node_index: usize,
    pub leaf_commitment: G1Affine,
    pub evaluation_point: Fr,
    pub evaluation_value: Fr,
    pub leaf_polynomial_coefficients: Vec<Fr>,
    pub leaf_opening_witness: G1Affine,
    pub path: Vec<VplusPublicPathStep>,
}


// ---------------------------------------------------------
// Public Parameters
//
// Verifier receives this.
// ---------------------------------------------------------

#[derive(Clone, Debug)]
pub struct VplusPublicParameters {
    pub srs: KzgSrs,
}


// ---------------------------------------------------------
// One Private Parent Step
//
// Prover only.
// ---------------------------------------------------------

#[derive(Clone, Debug)]
pub struct VplusPrivatePathStep {
    pub node_index: usize,
    pub record_position: usize,
    pub mapping: String,
    pub separator_key: i64,
    pub left_commitment: G1Affine,
    pub right_commitment: G1Affine,
    pub evaluation_point: Fr,
    pub evaluation_value: Fr,
    pub polynomial_coefficients: Vec<Fr>,
    pub parent_commitment: G1Affine,
    pub opening_witness: G1Affine,
}


// ---------------------------------------------------------
// Complete Private Membership Witness
//
// Prover only.
// ---------------------------------------------------------

#[derive(Clone, Debug)]
pub struct VplusPrivateWitness {
    pub key: i64,
    pub value: [u8; 32],
    pub leaf_node_index: usize,
    pub global_index: usize,
    pub digest: [u8; 32],
    pub y_low: Fr,
    pub y_high: Fr,
    pub leaf_evaluation_point: Fr,
    pub leaf_evaluation_value: Fr,
    pub leaf_polynomial_coefficients: Vec<Fr>,
    pub leaf_commitment: G1Affine,
    pub leaf_opening_witness: G1Affine,
    pub path: Vec<VplusPrivatePathStep>,
}


// ---------------------------------------------------------
// Loaded Membership Case
// ---------------------------------------------------------

#[derive(Clone, Debug)]
pub struct LoadedMembership {
    pub source_kzg_file: String,
    pub source_tree_file: String,
    pub public_statement: VplusPublicStatement,
    pub public_parameters: VplusPublicParameters,
    pub private_witness: VplusPrivateWitness,
    pub public_values: Vec<BabyBear>,
    pub private_values: Vec<BabyBear>,
}


// ---------------------------------------------------------
// Extract Decimal Integers From Arkworks Debug Strings
// ---------------------------------------------------------

fn extractDecimalNumbers(text: &str) -> Vec<String> {
    let mut numbers = Vec::new();
    let mut current = String::new();

    for character in text.chars() {
        if character.is_ascii_digit() {
            current.push(character);
        } else if !current.is_empty() {
            numbers.push(current.clone());
            current.clear();
        }
    }

    if !current.is_empty() {
        numbers.push(current);
    }

    numbers
}


// ---------------------------------------------------------
// Parse BN254 G1 Point
// ---------------------------------------------------------

fn parseG1(text: &str) -> G1Affine {
    let numbers = extractDecimalNumbers(text);

    assert_eq!(
        numbers.len(),
        2,
        "Could not parse G1 point: {}",
        text
    );

    let x = Fq::from_str(&numbers[0])
        .expect("Could not parse G1 x coordinate");

    let y = Fq::from_str(&numbers[1])
        .expect("Could not parse G1 y coordinate");

    G1Affine::new_unchecked(x, y)
}


// ---------------------------------------------------------
// Parse BN254 G2 Point
// ---------------------------------------------------------

fn parseG2(text: &str) -> G2Affine {
    let numbers = extractDecimalNumbers(text);

    assert_eq!(
        numbers.len(),
        4,
        "Could not parse G2 point: {}",
        text
    );

    let x_c0 = Fq::from_str(&numbers[0])
        .expect("Could not parse G2 x.c0");

    let x_c1 = Fq::from_str(&numbers[1])
        .expect("Could not parse G2 x.c1");

    let y_c0 = Fq::from_str(&numbers[2])
        .expect("Could not parse G2 y.c0");

    let y_c1 = Fq::from_str(&numbers[3])
        .expect("Could not parse G2 y.c1");

    let x = Fq2::new(x_c0, x_c1);
    let y = Fq2::new(y_c0, y_c1);

    G2Affine::new_unchecked(x, y)
}


// ---------------------------------------------------------
// Parse BN254 Fr
// ---------------------------------------------------------

fn parseFr(text: &str) -> Fr {
    Fr::from_str(text)
        .expect("Could not parse BN254 scalar")
}


// ---------------------------------------------------------
// Parse Polynomial Coefficients
// ---------------------------------------------------------

fn parsePolynomialCoefficients(node: &Value) -> Vec<Fr> {
    match node.get("polynomial_coefficients") {
        Some(value) => value
            .as_array()
            .expect("polynomial_coefficients must be an array")
            .iter()
            .map(|coefficient| {
                parseFr(
                    coefficient
                        .as_str()
                        .expect("Polynomial coefficient must be a string")
                )
            })
            .collect(),

        None => {
            let records = node["records"]
                .as_array()
                .expect("Node records missing");

            let mut points = Vec::new();

            for record in records {
                let x = parseFr(
                    record["evaluation_point"]
                        .as_str()
                        .expect("evaluation_point missing")
                );

                let y = parseFr(
                    record["evaluation_value"]
                        .as_str()
                        .expect("evaluation_value missing")
                );

                points.push((x, y));
            }

            lagrangeInterpolate(&points)
        }
    }
}


// ---------------------------------------------------------
// Find Tree Node
// ---------------------------------------------------------

fn findTreeNode<'a>(
    tree_json: &'a Value,
    node_index: usize
) -> &'a Value {
    tree_json["nodes"]
        .as_array()
        .expect("Tree nodes missing")
        .iter()
        .find(|node| {
            node["node_index"]
                .as_u64()
                .expect("node_index missing")
                as usize
                == node_index
        })
        .expect("Tree node not found")
}


// ---------------------------------------------------------
// Find KZG Node
// ---------------------------------------------------------

fn findKzgNode<'a>(
    kzg_json: &'a Value,
    node_index: usize
) -> &'a Value {
    let root = &kzg_json["root"];

    if root["node_index"]
        .as_u64()
        .expect("Root node_index missing")
        as usize
        == node_index
    {
        return root;
    }

    if let Some(parents) = kzg_json["parents"].as_array() {
        if let Some(node) = parents.iter().find(|node| {
            node["node_index"]
                .as_u64()
                .expect("Parent node_index missing")
                as usize
                == node_index
        }) {
            return node;
        }
    }

    if let Some(leaves) = kzg_json["leaves"].as_array() {
        if let Some(node) = leaves.iter().find(|node| {
            node["node_index"]
                .as_u64()
                .expect("Leaf node_index missing")
                as usize
                == node_index
        }) {
            return node;
        }
    }

    panic!("KZG node {} not found", node_index);
}


// ---------------------------------------------------------
// Build Public KZG SRS From JSON
// ---------------------------------------------------------

fn parseSrs(kzg_json: &Value) -> KzgSrs {
    let srs_json = &kzg_json["SRS"];

    let g1_powers = srs_json["G1_powers"]
        .as_array()
        .expect("SRS G1_powers missing")
        .iter()
        .map(|point| {
            parseG1(
                point
                    .as_str()
                    .expect("G1 power must be a string")
            )
        })
        .collect();

    let g2 = parseG2(
        srs_json["G2"]
            .as_str()
            .expect("SRS G2 missing")
    );

    let tau_g2 = parseG2(
        srs_json["tau_G2"]
            .as_str()
            .expect("SRS tau_G2 missing")
    );

    let max_degree = srs_json["max_degree"]
        .as_u64()
        .expect("SRS max_degree missing")
        as usize;

    KzgSrs {
        g1_powers,
        g2,
        tau_g2,
        max_degree,
    }
}


// ---------------------------------------------------------
// Generate Same 32-Byte Leaf Value As Existing main.rs
// ---------------------------------------------------------

fn generateValue(key: i64) -> [u8; 32] {
    let input = format!("Value{}", key);
    let digest = Sha256::digest(input.as_bytes());

    let mut value = [0u8; 32];
    value.copy_from_slice(&digest);

    value
}


// ---------------------------------------------------------
// Choose One KZG JSON File
// ---------------------------------------------------------

fn chooseRandomKzgFile(folder: &Path) -> PathBuf {
    let mut files = Vec::new();

    for entry in fs::read_dir(folder)
        .expect("Could not read JSON folder")
    {
        let entry = entry
            .expect("Could not read folder entry");

        let path = entry.path();

        let name = path
            .file_name()
            .expect("File name missing")
            .to_string_lossy();

        if name.starts_with("kzg_details_keys")
            && name.ends_with(".json")
        {
            files.push(path);
        }
    }

    files.sort();

    assert!(
        !files.is_empty(),
        "No kzg_details_keys*.json files found"
    );

    let index = rand::random_range(0..files.len());

    files[index].clone()
}


// ---------------------------------------------------------
// Matching Tree JSON File
// ---------------------------------------------------------

fn matchingTreeFile(kzg_file: &Path) -> PathBuf {
    let file_name = kzg_file
        .file_name()
        .expect("KZG file name missing")
        .to_string_lossy()
        .to_string();

    let tree_file_name = file_name.replacen(
        "kzg_details_",
        "vplus_tree_",
        1
    );

    kzg_file.with_file_name(tree_file_name)
}


// ---------------------------------------------------------
// Choose One Random Existing Key
// ---------------------------------------------------------

fn chooseRandomLeafOpening<'a>(
    kzg_json: &'a Value
) -> (&'a Value, &'a Value) {
    let leaves = kzg_json["leaves"]
        .as_array()
        .expect("KZG leaves missing");

    let mut candidates = Vec::new();

    for (leaf_index, leaf) in leaves.iter().enumerate() {
        let openings = leaf["openings"]
            .as_array()
            .expect("Leaf openings missing");

        for (opening_index, opening) in openings.iter().enumerate() {
            if opening["verification"]
                .as_bool()
                .unwrap_or(false)
            {
                candidates.push((leaf_index, opening_index));
            }
        }
    }

    assert!(
        !candidates.is_empty(),
        "No verified leaf openings found"
    );

    let selected = rand::random_range(0..candidates.len());
    let (leaf_index, opening_index) = candidates[selected];

    let leaf = &leaves[leaf_index];

    let opening = &leaf["openings"]
        .as_array()
        .expect("Leaf openings missing")[opening_index];

    (leaf, opening)
}


// ---------------------------------------------------------
// Load And Verify One Random Membership
// ---------------------------------------------------------

pub fn loadRandomMembershipFromJson(
    folder: &str
) -> LoadedMembership {
    let folder = Path::new(folder);

    let kzg_file = chooseRandomKzgFile(folder);
    let tree_file = matchingTreeFile(&kzg_file);

    assert!(
        tree_file.exists(),
        "Matching tree JSON file not found: {:?}",
        tree_file
    );

    println!();
    println!("==========================================");
    println!("Load V+ Tree Membership From JSON");
    println!("==========================================");
    println!("KZG JSON  : {}", kzg_file.display());
    println!("Tree JSON : {}", tree_file.display());

    let kzg_text = fs::read_to_string(&kzg_file)
        .expect("Could not read KZG JSON");

    let tree_text = fs::read_to_string(&tree_file)
        .expect("Could not read tree JSON");

    let kzg_json: Value = serde_json::from_str(&kzg_text)
        .expect("Could not parse KZG JSON");

    let tree_json: Value = serde_json::from_str(&tree_text)
        .expect("Could not parse tree JSON");

    let srs = parseSrs(&kzg_json);

    let (leaf_json, opening_json) =
        chooseRandomLeafOpening(&kzg_json);

    let key = opening_json["key"]
        .as_i64()
        .expect("Selected key missing");

    let leaf_node_index = leaf_json["node_index"]
        .as_u64()
        .expect("Leaf node_index missing")
        as usize;

    let global_index = opening_json["global_index"]
        .as_u64()
        .expect("global_index missing")
        as usize;

    let leaf_evaluation_point = parseFr(
        opening_json["evaluation_point"]
            .as_str()
            .expect("Leaf evaluation_point missing")
    );

    let leaf_evaluation_value = parseFr(
        opening_json["evaluation_value"]
            .as_str()
            .expect("Leaf evaluation_value missing")
    );

    let leaf_opening_witness = parseG1(
        opening_json["witness"]
            .as_str()
            .expect("Leaf witness missing")
    );

    let leaf_commitment = parseG1(
        leaf_json["commitment"]
            .as_str()
            .expect("Leaf commitment missing")
    );

    let leaf_polynomial_coefficients =
        parsePolynomialCoefficients(leaf_json);

    // -----------------------------------------------------
    // Rebuild Hidden Leaf Value And Digest
    // -----------------------------------------------------

    let value = generateValue(key);

    let digest = hashKeyValue(
        key,
        &value
    );

    let (y_low, y_high) =
        digestToFieldLimbs(&digest);

    let expected_leaf_value =
        y_low + y_high;

    assert_eq!(
        leaf_evaluation_value,
        expected_leaf_value,
        "Leaf evaluation value does not match hidden key/value digest"
    );

    assert_eq!(
        leaf_evaluation_point,
        Fr::from(global_index as u64),
        "Leaf evaluation point does not match global index"
    );

    let evaluated_leaf_value = evaluatePolynomial(
        &leaf_polynomial_coefficients,
        leaf_evaluation_point
    );

    assert_eq!(
        evaluated_leaf_value,
        leaf_evaluation_value,
        "Leaf polynomial evaluation does not match JSON opening"
    );

    let recomputed_leaf_commitment = commitPolynomial(
        &leaf_polynomial_coefficients,
        &srs
    );

    assert_eq!(
        recomputed_leaf_commitment,
        leaf_commitment,
        "Leaf commitment does not match polynomial coefficients"
    );

    let leaf_verification = verifyEval(
        &leaf_commitment,
        leaf_evaluation_point,
        leaf_evaluation_value,
        &leaf_opening_witness,
        &srs
    );

    assert!(
        leaf_verification,
        "Selected leaf KZG opening failed"
    );

    println!();
    println!("Selected Existing Key : {}", key);
    println!("Leaf Node             : {}", leaf_node_index);
    println!("Global Index          : {}", global_index);
    println!("Leaf KZG Verification : PASS");

    // -----------------------------------------------------
    // Trusted Public Root
    // -----------------------------------------------------

    let root_node_index = tree_json["root_node"]
        .as_u64()
        .expect("root_node missing")
        as usize;

    let root_commitment = parseG1(
        kzg_json["root"]["commitment"]
            .as_str()
            .expect("Root commitment missing")
    );

    // -----------------------------------------------------
    // Reconstruct And Verify Leaf -> Root Path
    // -----------------------------------------------------

    let mut path = Vec::new();

    let mut current_node_index =
        leaf_node_index;

    let mut verified_child_commitment =
        leaf_commitment;

    while current_node_index != root_node_index {
        let current_tree_node = findTreeNode(
            &tree_json,
            current_node_index
        );

        let parent_node_index = current_tree_node["parent_node"]
            .as_u64()
            .expect("Parent node missing")
            as usize;

        let parent_tree_node = findTreeNode(
            &tree_json,
            parent_node_index
        );

        let parent_children = parent_tree_node["children"]
            .as_array()
            .expect("Parent children missing");

        let child_position = parent_children
            .iter()
            .position(|child| {
                child
                    .as_u64()
                    .expect("Child index missing")
                    as usize
                    == current_node_index
            })
            .expect("Current child not found in parent");

        let parent_keys = parent_tree_node["keys"]
            .as_array()
            .expect("Parent keys missing");

        let (record_position, mapping) =
            if child_position < parent_keys.len() {
                (child_position, "LPC")
            } else {
                (parent_keys.len() - 1, "RPC")
            };

        let parent_kzg_node = findKzgNode(
            &kzg_json,
            parent_node_index
        );

        let record = &parent_kzg_node["records"]
            .as_array()
            .expect("Parent records missing")[record_position];

        let separator_key = record["key"]
            .as_i64()
            .expect("Internal separator key missing");

        let left_commitment = parseG1(
            record["LPC"]
                .as_str()
                .expect("LPC missing")
        );

        let right_commitment = parseG1(
            record["RPC"]
                .as_str()
                .expect("RPC missing")
        );

        let authenticated_child_commitment =
            if mapping == "LPC" {
                left_commitment
            } else {
                right_commitment
            };

        assert_eq!(
            authenticated_child_commitment,
            verified_child_commitment,
            "Child-to-parent commitment linkage failed at node {}",
            parent_node_index
        );

        let evaluation_point = parseFr(
            record["evaluation_point"]
                .as_str()
                .expect("Internal evaluation_point missing")
        );

        let evaluation_value = parseFr(
            record["evaluation_value"]
                .as_str()
                .expect("Internal evaluation_value missing")
        );

        let expected_internal_value = internalRecordEvaluation(
            separator_key,
            &left_commitment,
            &right_commitment
        );

        assert_eq!(
            evaluation_value,
            expected_internal_value,
            "Internal record hash/evaluation mismatch at node {}",
            parent_node_index
        );

        let parent_commitment = parseG1(
            parent_kzg_node["commitment"]
                .as_str()
                .expect("Parent commitment missing")
        );

        let opening_witness = parseG1(
            record["witness"]
                .as_str()
                .expect("Internal witness missing")
        );

        let polynomial_coefficients =
            parsePolynomialCoefficients(parent_kzg_node);

        let polynomial_value = evaluatePolynomial(
            &polynomial_coefficients,
            evaluation_point
        );

        assert_eq!(
            polynomial_value,
            evaluation_value,
            "Parent polynomial evaluation mismatch at node {}",
            parent_node_index
        );

        let recomputed_parent_commitment = commitPolynomial(
            &polynomial_coefficients,
            &srs
        );

        assert_eq!(
            recomputed_parent_commitment,
            parent_commitment,
            "Parent commitment does not match polynomial coefficients at node {}",
            parent_node_index
        );

        let parent_verification = verifyEval(
            &parent_commitment,
            evaluation_point,
            evaluation_value,
            &opening_witness,
            &srs
        );

        assert!(
            parent_verification,
            "Parent KZG verification failed at node {}",
            parent_node_index
        );

        println!(
            "Parent Node {} KZG Verification : PASS",
            parent_node_index
        );

        path.push(
            VplusPrivatePathStep {
                node_index: parent_node_index,
                record_position,
                mapping: mapping.to_string(),
                separator_key,
                left_commitment,
                right_commitment,
                evaluation_point,
                evaluation_value,
                polynomial_coefficients,
                parent_commitment,
                opening_witness,
            }
        );

        verified_child_commitment =
            parent_commitment;

        current_node_index =
            parent_node_index;
    }

    assert_eq!(
        verified_child_commitment,
        root_commitment,
        "Membership path does not terminate at trusted public root"
    );

    println!();
    println!("Trusted Root Verification : PASS");
    println!("Complete Membership        : PASS");

    // -----------------------------------------------------
    // Build Verifier-Public Query Path
    //
    // Only the selected queried key's leaf -> root path is public.
    // The selected leaf polynomial coefficients and selected leaf KZG
    // opening witness are public. Internal-node coefficients and
    // internal-node KZG opening witnesses remain private.
    // -----------------------------------------------------

    let public_path: Vec<VplusPublicPathStep> =
        path
            .iter()
            .map(
                |step|
                {
                    VplusPublicPathStep {
                        node_index: step.node_index,
                        record_position: step.record_position,
                        mapping: step.mapping.clone(),
                        separator_key: step.separator_key,
                        left_commitment: step.left_commitment,
                        right_commitment: step.right_commitment,
                        evaluation_point: step.evaluation_point,
                        evaluation_value: step.evaluation_value,
                        parent_commitment: step.parent_commitment,
                    }
                }
            )
            .collect();


    let public_statement =
        VplusPublicStatement {
            key,
            root_commitment,
            leaf_node_index,
            leaf_commitment,
            evaluation_point: leaf_evaluation_point,
            evaluation_value: leaf_evaluation_value,
            leaf_polynomial_coefficients: leaf_polynomial_coefficients.clone(),
            leaf_opening_witness,
            path: public_path,
        };

    let public_parameters =
        VplusPublicParameters {
            srs: srs.clone(),
        };

    let private_witness =
        VplusPrivateWitness {
            key,
            value,
            leaf_node_index,
            global_index,
            digest,
            y_low,
            y_high,
            leaf_evaluation_point,
            leaf_evaluation_value,
            leaf_polynomial_coefficients,
            leaf_commitment,
            leaf_opening_witness,
            path,
        };

    let mut public_values = encodePublicStatement(
        &public_statement
    );

    // Public SRS is part of the verifier statement and is
    // passed to both prove(...) and verify(...).
    public_values.extend(
        encodePublicSrs(
            &srs
        )
    );

    let private_values = encodePrivateWitness(
        &private_witness
    );

    println!();
    println!("==========================================");
    println!("STARK Input Separation");
    println!("==========================================");
    println!("Verifier Public Key       : {}", public_statement.key);
    println!("Verifier Public Root      : {:?}", public_statement.root_commitment);
    println!("Verifier Evaluation Point : {:?}", public_statement.evaluation_point);
    println!("Verifier Evaluation Value : {:?}", public_statement.evaluation_value);
    println!("Verifier Leaf Coefficients: {}", public_statement.leaf_polynomial_coefficients.len());
    println!("Verifier Selected Witness : {:?}", public_statement.leaf_opening_witness);
    println!("Verifier Query Path Steps : {}", public_statement.path.len());
    println!("Verifier Public SRS       : YES");
    println!("Verifier Leaf Coefficients: YES");
    println!("Verifier Internal Coeffs  : NO");
    println!("Verifier y_low / y_high   : NO");
    println!("Verifier Internal Witnesses: NO");
    println!("Verifier Private Trace    : NO");
    println!("Public BabyBear Elements  : {}", public_values.len());
    println!("Private BabyBear Elements : {}", private_values.len());

    LoadedMembership {
        source_kzg_file: kzg_file.display().to_string(),
        source_tree_file: tree_file.display().to_string(),
        public_statement,
        public_parameters,
        private_witness,
        public_values,
        private_values,
    }
}


// ---------------------------------------------------------
// Push One Byte Into BabyBear
// ---------------------------------------------------------

fn pushByte(
    output: &mut Vec<BabyBear>,
    value: u8
) {
    output.push(
        BabyBear::from_u32(
            value as u32
        )
    );
}


// ---------------------------------------------------------
// Push Raw Bytes Into BabyBear
// ---------------------------------------------------------

fn pushBytes(
    output: &mut Vec<BabyBear>,
    bytes: &[u8]
) {
    for byte in bytes {
        pushByte(
            output,
            *byte
        );
    }
}


// ---------------------------------------------------------
// Push usize
// ---------------------------------------------------------

fn pushUsize(
    output: &mut Vec<BabyBear>,
    value: usize
) {
    pushBytes(
        output,
        &(value as u64).to_be_bytes()
    );
}


// ---------------------------------------------------------
// Push i64
// ---------------------------------------------------------

fn pushI64(
    output: &mut Vec<BabyBear>,
    value: i64
) {
    pushBytes(
        output,
        &value.to_be_bytes()
    );
}


// ---------------------------------------------------------
// Push Canonical Arkworks Value
// ---------------------------------------------------------

fn pushCanonical<T>(
    output: &mut Vec<BabyBear>,
    value: &T
)
where
    T: CanonicalSerialize,
{
    let mut bytes = Vec::new();

    value
        .serialize_compressed(&mut bytes)
        .expect("Could not serialize private/public value");

    pushBytes(
        output,
        &bytes
    );
}


// ---------------------------------------------------------
// Public Encoding
//
// Verifier receives:
//
//     selected key
//     trusted root commitment
//     selected leaf commitment
//     selected leaf evaluation point
//     selected leaf evaluation value
//     selected leaf polynomial coefficients
//     selected leaf KZG opening witness
//     selected leaf -> root path data only
//
// NEVER encoded here:
//
//     internal-node polynomial coefficients
//     internal-node KZG opening witnesses
//     unrelated leaf/node data
//     mapped values
//     y_low
//     y_high
//     private trace
// ---------------------------------------------------------

pub fn encodePublicStatement(
    statement: &VplusPublicStatement
) -> Vec<BabyBear> {
    let mut values = Vec::new();

    pushI64(
        &mut values,
        statement.key
    );

    pushCanonical(
        &mut values,
        &statement.root_commitment
    );

    pushUsize(
        &mut values,
        statement.leaf_node_index
    );

    pushCanonical(
        &mut values,
        &statement.leaf_commitment
    );

    pushCanonical(
        &mut values,
        &statement.evaluation_point
    );

    pushCanonical(
        &mut values,
        &statement.evaluation_value
    );

    pushUsize(
        &mut values,
        statement.leaf_polynomial_coefficients.len()
    );

    for coefficient in &statement.leaf_polynomial_coefficients {
        pushCanonical(
            &mut values,
            coefficient
        );
    }

    pushCanonical(
        &mut values,
        &statement.leaf_opening_witness
    );

    pushUsize(
        &mut values,
        statement.path.len()
    );

    for step in &statement.path {
        pushUsize(
            &mut values,
            step.node_index
        );

        pushUsize(
            &mut values,
            step.record_position
        );

        pushByte(
            &mut values,
            if step.mapping == "LPC" {
                0
            } else {
                1
            }
        );

        pushI64(
            &mut values,
            step.separator_key
        );

        pushCanonical(
            &mut values,
            &step.left_commitment
        );

        pushCanonical(
            &mut values,
            &step.right_commitment
        );

        pushCanonical(
            &mut values,
            &step.evaluation_point
        );

        pushCanonical(
            &mut values,
            &step.evaluation_value
        );

        pushCanonical(
            &mut values,
            &step.parent_commitment
        );
    }

    values
}


// ---------------------------------------------------------
// Private Prover Encoding
// ---------------------------------------------------------

pub fn encodePrivateWitness(
    witness: &VplusPrivateWitness
) -> Vec<BabyBear> {
    let mut values = Vec::new();

    pushBytes(
        &mut values,
        &witness.value
    );

    pushUsize(
        &mut values,
        witness.leaf_node_index
    );

    pushUsize(
        &mut values,
        witness.global_index
    );

    pushBytes(
        &mut values,
        &witness.digest
    );

    pushCanonical(
        &mut values,
        &witness.y_low
    );

    pushCanonical(
        &mut values,
        &witness.y_high
    );

    pushCanonical(
        &mut values,
        &witness.leaf_evaluation_point
    );

    pushCanonical(
        &mut values,
        &witness.leaf_evaluation_value
    );

    pushUsize(
        &mut values,
        witness.leaf_polynomial_coefficients.len()
    );

    for coefficient in &witness.leaf_polynomial_coefficients {
        pushCanonical(
            &mut values,
            coefficient
        );
    }

    pushCanonical(
        &mut values,
        &witness.leaf_commitment
    );

    pushCanonical(
        &mut values,
        &witness.leaf_opening_witness
    );

    pushUsize(
        &mut values,
        witness.path.len()
    );

    for step in &witness.path {
        pushUsize(
            &mut values,
            step.node_index
        );

        pushUsize(
            &mut values,
            step.record_position
        );

        pushByte(
            &mut values,
            if step.mapping == "LPC" {
                0
            } else {
                1
            }
        );

        pushI64(
            &mut values,
            step.separator_key
        );

        pushCanonical(
            &mut values,
            &step.left_commitment
        );

        pushCanonical(
            &mut values,
            &step.right_commitment
        );

        pushCanonical(
            &mut values,
            &step.evaluation_point
        );

        pushCanonical(
            &mut values,
            &step.evaluation_value
        );

        pushUsize(
            &mut values,
            step.polynomial_coefficients.len()
        );

        for coefficient in &step.polynomial_coefficients {
            pushCanonical(
                &mut values,
                coefficient
            );
        }

        pushCanonical(
            &mut values,
            &step.parent_commitment
        );

        pushCanonical(
            &mut values,
            &step.opening_witness
        );
    }

    values
}


// ---------------------------------------------------------
// Selected-Key STARK Trace Layout
//
// ONE selected key.
// ONE recursive leaf -> root trace.
// ONE AIR.
// ONE Plonky3 configuration.
// ONE prove(...).
// ONE proof.
// ONE verify(...).
//
// Public values are NOT hidden inside the AIR struct. They
// are passed explicitly to prove(...) and verify(...), and
// the AIR binds them through builder.public_values().
// ---------------------------------------------------------

const TRACE_ACTIVE: usize = 0;
const TRACE_LEVEL: usize = 1;
const TRACE_MAPPING: usize = 2;

const TRACE_CURRENT_COMMITMENT: usize = 3;
const TRACE_LEFT_COMMITMENT: usize =
    TRACE_CURRENT_COMMITMENT + 32;
const TRACE_RIGHT_COMMITMENT: usize =
    TRACE_LEFT_COMMITMENT + 32;
const TRACE_PARENT_COMMITMENT: usize =
    TRACE_RIGHT_COMMITMENT + 32;
const TRACE_PATH_EVALUATION_POINT: usize =
    TRACE_PARENT_COMMITMENT + 32;
const TRACE_PATH_EVALUATION_VALUE: usize =
    TRACE_PATH_EVALUATION_POINT + 32;

// Every verifier-public BabyBear value is mirrored on the
// first row and constrained equal to builder.public_values().
const TRACE_PUBLIC_VALUES_START: usize =
    TRACE_PATH_EVALUATION_VALUE + 32;


// ---------------------------------------------------------
// Public Statement Layout
//
// This mirrors encodePublicStatement(...).
// ---------------------------------------------------------

#[derive(Clone, Debug)]
struct VplusPublicLayout {
    root_commitment_start: usize,
    leaf_commitment_start: usize,
    leaf_evaluation_point_start: usize,
    leaf_evaluation_value_start: usize,
    leaf_coefficients_start: usize,
    leaf_opening_witness_start: usize,
    path_start: usize,
    path_step_size: usize,
    statement_len: usize,
    total_public_values: usize,
}


fn buildPublicLayout(
    loaded: &LoadedMembership
) -> VplusPublicLayout {
    // encodePublicStatement:
    //
    // key                     8
    // root commitment        32
    // leaf node index         8
    // leaf commitment        32
    // evaluation point       32
    // evaluation value       32
    // coefficient count       8
    // coefficients          32*N
    // leaf witness            32
    // path count               8
    // each path step          185
    //
    // Public SRS is appended after encodePublicStatement.

    let root_commitment_start = 8;
    let leaf_commitment_start =
        root_commitment_start + 32 + 8;

    let leaf_evaluation_point_start =
        leaf_commitment_start + 32;

    let leaf_evaluation_value_start =
        leaf_evaluation_point_start + 32;

    let leaf_coefficients_start =
        leaf_evaluation_value_start + 32 + 8;

    let leaf_opening_witness_start =
        leaf_coefficients_start
        + loaded
            .public_statement
            .leaf_polynomial_coefficients
            .len()
            * 32;

    let path_count_start =
        leaf_opening_witness_start + 32;

    let path_start =
        path_count_start + 8;

    // node index       8
    // record position  8
    // mapping          1
    // separator key    8
    // LPC             32
    // RPC             32
    // eval point      32
    // eval value      32
    // parent          32
    let path_step_size =
        8 + 8 + 1 + 8 + 32 + 32 + 32 + 32 + 32;

    let statement_len =
        path_start
        + loaded.public_statement.path.len()
            * path_step_size;

    assert!(
        loaded.public_values.len() >= statement_len,
        "Public value vector is shorter than encoded public statement"
    );

    VplusPublicLayout {
        root_commitment_start,
        leaf_commitment_start,
        leaf_evaluation_point_start,
        leaf_evaluation_value_start,
        leaf_coefficients_start,
        leaf_opening_witness_start,
        path_start,
        path_step_size,
        statement_len,
        total_public_values:
            loaded.public_values.len(),
    }
}


fn publicPathStepStart(
    layout: &VplusPublicLayout,
    path_index: usize
) -> usize {
    layout.path_start
        + path_index * layout.path_step_size
}


// ---------------------------------------------------------
// Encode Public SRS
//
// Appended to loaded.public_values by the JSON loader.
// ---------------------------------------------------------

fn encodePublicSrs(
    srs: &KzgSrs
) -> Vec<BabyBear> {
    let mut values =
        Vec::new();

    pushUsize(
        &mut values,
        srs.g1_powers.len()
    );

    for point in &srs.g1_powers {
        pushCanonical(
            &mut values,
            point
        );
    }

    pushCanonical(
        &mut values,
        &srs.g2
    );

    pushCanonical(
        &mut values,
        &srs.tau_g2
    );

    pushUsize(
        &mut values,
        srs.max_degree
    );

    values
}


// ---------------------------------------------------------
// V+ Membership AIR
//
// The AIR currently constrains:
//
// 1. verifier public values are exactly the values used by
//    the prover;
// 2. selected leaf commitment is the first current child;
// 3. each path row chooses LPC or RPC;
// 4. selected child equals the chosen LPC/RPC;
// 5. parent commitment recursively becomes the next child;
// 6. each row equals the exact selected public path;
// 7. final parent commitment equals the public root.
//
// The loader already performs the existing Arkworks
// polynomial evaluation and KZG pairing verification before
// this trace is built.
//
// IMPORTANT:
// The BN254 polynomial arithmetic and pairing computation are
// still native Arkworks prechecks. They are NOT falsely
// labelled as AIR constraints here.
// ---------------------------------------------------------

#[derive(Clone, Debug)]
pub struct VplusMembershipAir {
    width: usize,
    path_len: usize,
    public_values_len: usize,
    layout: VplusPublicLayout,
}


impl VplusMembershipAir {
    fn fromLoadedMembership(
        loaded: &LoadedMembership
    ) -> Self {
        assert!(
            !loaded.public_statement.path.is_empty(),
            "Selected membership path must contain at least one parent step"
        );

        let layout =
            buildPublicLayout(
                loaded
            );

        let width =
            TRACE_PUBLIC_VALUES_START
            + loaded.public_values.len();

        Self {
            width,
            path_len:
                loaded.public_statement.path.len(),
            public_values_len:
                loaded.public_values.len(),
            layout,
        }
    }
}


impl BaseAir<BabyBear>
    for VplusMembershipAir
{
    fn width(
        &self
    ) -> usize {
        self.width
    }

    fn num_public_values(
        &self
    ) -> usize {
        self.public_values_len
    }

    fn main_next_row_columns(
        &self
    ) -> Vec<usize> {
        let mut columns =
            vec![
                TRACE_ACTIVE,
                TRACE_LEVEL,
            ];

        columns.extend(
            TRACE_CURRENT_COMMITMENT
                ..TRACE_CURRENT_COMMITMENT + 32
        );

        columns
    }
}


impl<AB> Air<AB>
    for VplusMembershipAir
where
    AB: AirBuilder<F = BabyBear>,
    AB::Expr: Clone,
{
    fn eval(
        &self,
        builder: &mut AB
    ) {
        let main =
            builder.main();

        let row =
            main.current_slice();

        let next =
            main.next_slice();

        // Copy the public input slice so the mutable builder
        // can be used below without borrow conflicts.
        let public_values =
            builder
                .public_values()
                .to_vec();

        let active =
            row[TRACE_ACTIVE];

        let level =
            row[TRACE_LEVEL];

        let mapping =
            row[TRACE_MAPPING];

        let next_active =
            next[TRACE_ACTIVE];

        let next_level =
            next[TRACE_LEVEL];

        builder.assert_bool(active);
        builder.assert_bool(mapping);

        builder
            .when_first_row()
            .assert_one(active);

        builder
            .when_first_row()
            .assert_zero(level);

        builder
            .when_last_row()
            .assert_zero(active);

        // -------------------------------------------------
        // Bind every verifier-public value into this proof.
        // -------------------------------------------------

        for i in 0..self.public_values_len {
            builder
                .when_first_row()
                .assert_eq(
                    row[
                        TRACE_PUBLIC_VALUES_START + i
                    ],
                    public_values[i]
                );
        }

        // -------------------------------------------------
        // Active rows are contiguous.
        // -------------------------------------------------

        builder
            .when_transition()
            .assert_zero(
                next_active
                    * (
                        active
                        - BabyBear::from_u32(1)
                    )
            );

        // -------------------------------------------------
        // Recursive path levels increment exactly by one.
        // -------------------------------------------------

        builder
            .when_transition()
            .assert_zero(
                next_active
                    * (
                        next_level
                        - level
                        - BabyBear::from_u32(1)
                    )
            );

        // -------------------------------------------------
        // Every active row must be a valid path level.
        // -------------------------------------------------

        let mut valid_level:
            AB::Expr =
            active.into();

        for i in 0..self.path_len {
            valid_level =
                valid_level
                * (
                    level
                    - BabyBear::from_u32(
                        i as u32
                    )
                );
        }

        builder.assert_zero(
            valid_level
        );

        // -------------------------------------------------
        // First current commitment = public selected leaf.
        // -------------------------------------------------

        for i in 0..32 {
            builder
                .when_first_row()
                .assert_eq(
                    row[
                        TRACE_CURRENT_COMMITMENT + i
                    ],
                    public_values[
                        self.layout
                            .leaf_commitment_start
                        + i
                    ]
                );
        }

        // -------------------------------------------------
        // LPC / RPC selection:
        //
        // mapping = 0 -> current = LPC
        // mapping = 1 -> current = RPC
        // -------------------------------------------------

        for i in 0..32 {
            let current =
                row[
                    TRACE_CURRENT_COMMITMENT + i
                ];

            let left =
                row[
                    TRACE_LEFT_COMMITMENT + i
                ];

            let right =
                row[
                    TRACE_RIGHT_COMMITMENT + i
                ];

            builder.assert_zero(
                active
                    * (
                        current
                        - left
                        - mapping
                            * (
                                right
                                - left
                            )
                    )
            );
        }

        // -------------------------------------------------
        // Parent result recursively becomes next child.
        // -------------------------------------------------

        for i in 0..32 {
            builder
                .when_transition()
                .assert_zero(
                    next_active
                        * (
                            next[
                                TRACE_CURRENT_COMMITMENT + i
                            ]
                            - row[
                                TRACE_PARENT_COMMITMENT + i
                            ]
                        )
                );
        }

        // -------------------------------------------------
        // Bind every path row to the exact selected
        // verifier-public key path.
        // -------------------------------------------------

        for path_index in 0..self.path_len {
            let mut selector:
                AB::Expr =
                active.into();

            for other_index in 0..self.path_len {
                if other_index != path_index {
                    selector =
                        selector
                        * (
                            level
                            - BabyBear::from_u32(
                                other_index as u32
                            )
                        );
                }
            }

            let public_start =
                publicPathStepStart(
                    &self.layout,
                    path_index
                );

            // Path step byte layout:
            //
            // node index       +0  .. +7
            // record position  +8  .. +15
            // mapping          +16
            // separator key    +17 .. +24
            // LPC              +25 .. +56
            // RPC              +57 .. +88
            // eval point       +89 .. +120
            // eval value       +121.. +152
            // parent           +153.. +184

            let public_mapping =
                public_values[
                    public_start + 16
                ];

            builder.assert_zero(
                selector.clone()
                    * (
                        mapping
                        - public_mapping
                    )
            );

            for i in 0..32 {
                builder.assert_zero(
                    selector.clone()
                        * (
                            row[
                                TRACE_LEFT_COMMITMENT + i
                            ]
                            - public_values[
                                public_start + 25 + i
                            ]
                        )
                );

                builder.assert_zero(
                    selector.clone()
                        * (
                            row[
                                TRACE_RIGHT_COMMITMENT + i
                            ]
                            - public_values[
                                public_start + 57 + i
                            ]
                        )
                );

                builder.assert_zero(
                    selector.clone()
                        * (
                            row[
                                TRACE_PATH_EVALUATION_POINT + i
                            ]
                            - public_values[
                                public_start + 89 + i
                            ]
                        )
                );

                builder.assert_zero(
                    selector.clone()
                        * (
                            row[
                                TRACE_PATH_EVALUATION_VALUE + i
                            ]
                            - public_values[
                                public_start + 121 + i
                            ]
                        )
                );

                builder.assert_zero(
                    selector.clone()
                        * (
                            row[
                                TRACE_PARENT_COMMITMENT + i
                            ]
                            - public_values[
                                public_start + 153 + i
                            ]
                        )
                );
            }
        }

        // -------------------------------------------------
        // Last active row must be path_len - 1 and its
        // parent must equal the public trusted root.
        // -------------------------------------------------

        let last_level =
            BabyBear::from_u32(
                (self.path_len - 1) as u32
            );

        let end_filter =
            active
            * (
                next_active
                - BabyBear::from_u32(1)
            );

        builder
            .when_transition()
            .assert_zero(
                end_filter.clone()
                    * (
                        level
                        - last_level
                    )
            );

        for i in 0..32 {
            builder
                .when_transition()
                .assert_zero(
                    end_filter.clone()
                        * (
                            row[
                                TRACE_PARENT_COMMITMENT + i
                            ]
                            - public_values[
                                self.layout
                                    .root_commitment_start
                                + i
                            ]
                        )
                );
        }
    }
}


// ---------------------------------------------------------
// Build ONE RowMajorMatrix
// ---------------------------------------------------------

pub fn buildPrivateTrace(
    loaded: &LoadedMembership
) -> RowMajorMatrix<BabyBear> {
    let trace_start =
        Instant::now();

    let witness =
        &loaded.private_witness;

    let air =
        VplusMembershipAir::
            fromLoadedMembership(
                loaded
            );

    let trace_height =
        witness
            .path
            .len()
            .max(8)
            .next_power_of_two();

    let mut values =
        vec![
            BabyBear::from_u32(0);
            trace_height * air.width
        ];

    for (
        path_index,
        step
    ) in witness.path.iter().enumerate()
    {
        let row_start =
            path_index * air.width;

        values[
            row_start + TRACE_ACTIVE
        ] =
            BabyBear::from_u32(1);

        values[
            row_start + TRACE_LEVEL
        ] =
            BabyBear::from_u32(
                path_index as u32
            );

        values[
            row_start + TRACE_MAPPING
        ] =
            if step.mapping == "LPC" {
                BabyBear::from_u32(0)
            } else {
                BabyBear::from_u32(1)
            };

        let current_commitment =
            if path_index == 0 {
                witness.leaf_commitment
            } else {
                witness.path[
                    path_index - 1
                ]
                    .parent_commitment
            };

        let current =
            canonical32(
                &current_commitment
            );

        let left =
            canonical32(
                &step.left_commitment
            );

        let right =
            canonical32(
                &step.right_commitment
            );

        let parent =
            canonical32(
                &step.parent_commitment
            );

        let evaluation_point =
            canonical32(
                &step.evaluation_point
            );

        let evaluation_value =
            canonical32(
                &step.evaluation_value
            );

        for i in 0..32 {
            values[
                row_start
                + TRACE_CURRENT_COMMITMENT
                + i
            ] =
                current[i];

            values[
                row_start
                + TRACE_LEFT_COMMITMENT
                + i
            ] =
                left[i];

            values[
                row_start
                + TRACE_RIGHT_COMMITMENT
                + i
            ] =
                right[i];

            values[
                row_start
                + TRACE_PARENT_COMMITMENT
                + i
            ] =
                parent[i];

            values[
                row_start
                + TRACE_PATH_EVALUATION_POINT
                + i
            ] =
                evaluation_point[i];

            values[
                row_start
                + TRACE_PATH_EVALUATION_VALUE
                + i
            ] =
                evaluation_value[i];
        }
    }

    // Mirror every verifier-public BabyBear value on row 0.
    for (
        i,
        value
    ) in loaded
        .public_values
        .iter()
        .enumerate()
    {
        values[
            TRACE_PUBLIC_VALUES_START + i
        ] =
            *value;
    }

    let privateTrace =
        RowMajorMatrix::new(
            values,
            air.width
        );

    let trace_time =
        trace_start.elapsed();

    println!();
    println!("==========================================");
    println!("ROW MAJOR MATRIX");
    println!("==========================================");
    println!("Trace Width               : {}", air.width);
    println!("Trace Height              : {}", trace_height);
    println!("Selected Path Levels      : {}", witness.path.len());
    println!("Public Input Elements     : {}", loaded.public_values.len());
    println!("Trace Build Time          : {:?}", trace_time);

    for (
        row_index,
        row
    ) in privateTrace
        .values
        .chunks(air.width)
        .enumerate()
    {
        println!();
        println!("Row {}:", row_index);

        for (
            column_index,
            value
        ) in row.iter().enumerate()
        {
            println!(
                "  Column {:>4} = {}",
                column_index,
                value.as_canonical_u32()
            );
        }
    }

    privateTrace
}


// ---------------------------------------------------------
// Print Exact Statement / Relation
// ---------------------------------------------------------

fn printStatementRelation(
    loaded: &LoadedMembership
) {
    println!();
    println!("==========================================");
    println!("STATEMENT / RELATION");
    println!("==========================================");

    println!(
        "Selected Key              : {}",
        loaded.public_statement.key
    );

    println!(
        "Selected Leaf             : {}",
        loaded.public_statement.leaf_node_index
    );

    println!(
        "Path Levels               : {}",
        loaded.public_statement.path.len()
    );

    println!(
        "Evaluation Point          : {:?}",
        loaded.public_statement.evaluation_point
    );

    println!(
        "Evaluation Value          : {:?}",
        loaded.public_statement.evaluation_value
    );

    println!(
        "Polynomial Coefficients   : {:?}",
        loaded
            .public_statement
            .leaf_polynomial_coefficients
    );

    println!(
        "Selected KZG Witness      : {:?}",
        loaded.public_statement.leaf_opening_witness
    );

    println!(
        "Leaf Commitment           : {:?}",
        loaded.public_statement.leaf_commitment
    );

    println!(
        "Trusted Root              : {:?}",
        loaded.public_statement.root_commitment
    );

    println!("Public SRS                : YES");

    println!();
    println!("AIR relation currently proved:");
    println!("  1. public statement == prover-bound statement");
    println!("  2. first child == selected public leaf commitment");
    println!("  3. each level selects LPC or RPC from the public path");
    println!("  4. selected child == chosen LPC/RPC");
    println!("  5. parent(level) == child(level + 1)");
    println!("  6. final parent == trusted public root");

    println!();
    println!("Native Arkworks preconditions already checked before STARK:");
    println!("  - selected leaf polynomial evaluation");
    println!("  - selected leaf KZG opening");
    println!("  - every internal path polynomial evaluation");
    println!("  - every internal path KZG opening");

    println!();
    println!("BN254 pairing arithmetic inside AIR : NOT CLAIMED");
}


// ---------------------------------------------------------
// ONE V+ Tree STARK Pipeline
//
// JSON
//   -> selected key
//   -> BabyBear public/private representation
//   -> recursive leaf -> root trace
//   -> statement/relation
//   -> AIR
//   -> configuration
//   -> prove(..., public_values)
//   -> ONE proof
//   -> verify(..., public_values)
// ---------------------------------------------------------

pub fn runStarkProof(
    loaded: &LoadedMembership
) {
    let witness =
        &loaded.private_witness;

    println!();
    println!("==========================================");
    println!("JSON TREE DATA");
    println!("==========================================");
    println!("KZG JSON                  : {}", loaded.source_kzg_file);
    println!("Tree JSON                 : {}", loaded.source_tree_file);
    println!("Selected Existing Key     : {}", loaded.public_statement.key);
    println!("Selected Leaf Node        : {}", loaded.public_statement.leaf_node_index);
    println!("Selected Path Levels      : {}", loaded.public_statement.path.len());

    println!();
    println!("==========================================");
    println!("BABYBEAR CONVERSION");
    println!("==========================================");
    println!("Public BabyBear Elements  : {}", loaded.public_values.len());
    println!("Private BabyBear Elements : {}", loaded.private_values.len());
    println!("Public SRS Included       : YES");

    println!();
    println!("==========================================");
    println!("PROVER");
    println!("==========================================");
    println!("Selected Key              : {}", loaded.public_statement.key);
    println!("Selected KZG Witness      : {:?}", loaded.public_statement.leaf_opening_witness);
    println!("Evaluation Point          : {:?}", loaded.public_statement.evaluation_point);
    println!("Evaluation Value          : {:?}", loaded.public_statement.evaluation_value);
    println!("Polynomial Coefficients   : {:?}", loaded.public_statement.leaf_polynomial_coefficients);
    println!("Leaf Commitment           : {:?}", loaded.public_statement.leaf_commitment);
    println!("Trusted Root              : {:?}", loaded.public_statement.root_commitment);
    println!("Public SRS                : YES");

    println!();
    println!("Recursive Selected Path:");
    println!(
        "  Leaf {}",
        witness.leaf_node_index
    );

    for step in &witness.path {
        if step.parent_commitment
            == loaded.public_statement.root_commitment
        {
            println!(
                "    -> Node {} / ROOT",
                step.node_index
            );
        } else {
            println!(
                "    -> Node {}",
                step.node_index
            );
        }
    }

    let privateTrace =
        buildPrivateTrace(
            loaded
        );

    printStatementRelation(
        loaded
    );

    let air_start =
        Instant::now();

    let air =
        VplusMembershipAir::
            fromLoadedMembership(
                loaded
            );

    let air_time =
        air_start.elapsed();

    println!();
    println!("==========================================");
    println!("AIR");
    println!("==========================================");
    println!("AIR Name                  : VplusMembershipAir");
    println!("AIR Width                 : {}", air.width());
    println!("Public Values             : {}", air.num_public_values());
    println!("Selected Path Levels      : {}", air.path_len);
    println!("Public Values Bound       : YES");
    println!("Recursive LPC/RPC Link    : CONSTRAINED");
    println!("Parent -> Next Level      : CONSTRAINED");
    println!("Final Parent -> Root      : CONSTRAINED");
    println!("Native KZG Precheck       : COMPLETED");
    println!("AIR Construction Time     : {:?}", air_time);

    type Val =
        BabyBear;

    type Challenge =
        BinomialExtensionField<
            Val,
            4
        >;

    type ByteHash =
        P3Sha256;

    type FieldHash =
        SerializingHasher<
            ByteHash
        >;

    let byte_hash =
        ByteHash {};

    let field_hash =
        FieldHash::new(
            P3Sha256
        );

    type MyCompress =
        CompressionFunctionFromHasher<
            ByteHash,
            2,
            32
        >;

    let compress =
        MyCompress::new(
            byte_hash
        );

    type ValMmcs =
        MerkleTreeMmcs<
            Val,
            u8,
            FieldHash,
            MyCompress,
            2,
            32
        >;

    let val_mmcs =
        ValMmcs::new(
            field_hash,
            compress,
            3
        );

    type ChallengeMmcs =
        ExtensionMmcs<
            Val,
            Challenge,
            ValMmcs
        >;

    let challenge_mmcs =
        ChallengeMmcs::new(
            val_mmcs.clone()
        );

    type Dft =
        p3_dft::Radix2Bowers;

    let dft =
        Dft::default();

    type Challenger =
        SerializingChallenger32<
            Val,
            HashChallenger<
                u8,
                ByteHash,
                32
            >
        >;

    let challenger =
        Challenger::from_hasher(
            vec![],
            byte_hash
        );

    let fri_params =
        FriParameters::new_benchmark(
            challenge_mmcs
        );

    type Pcs =
        TwoAdicFriPcs<
            Val,
            Dft,
            ValMmcs,
            ChallengeMmcs
        >;

    let pcs =
        Pcs::new(
            dft,
            val_mmcs,
            fri_params
        );

    type MyConfig =
        StarkConfig<
            Pcs,
            Challenge,
            Challenger
        >;

    let config =
        MyConfig::new(
            pcs,
            challenger
        );

    println!();
    println!("==========================================");
    println!("CONFIGURATION");
    println!("==========================================");
    println!("Field                     : BabyBear");
    println!("Challenge Extension       : Degree 4");
    println!("PCS                       : TwoAdicFriPcs");
    println!("FRI                       : CONFIGURED");
    println!("Merkle MMCS               : CONFIGURED");
    println!("Challenger                : CONFIGURED");
    println!("Zero Knowledge            : NO — standard TwoAdicFriPcs");

    println!();
    println!("==========================================");
    println!("STARK PROOF");
    println!("==========================================");
    println!("Generating STARK Proof...");

    let proof_start =
        Instant::now();

    let proof =
        prove(
            &config,
            &air,
            privateTrace,
            &loaded.public_values
        )
        .expect(
            "STARK proof generation failed"
        );

    let proof_time =
        proof_start.elapsed();

    println!("STARK Proof Generated     : YES");
    println!("STARK Proof Count         : 1");
    println!("Path Levels In Proof      : {}", loaded.public_statement.path.len());
    println!("Proof Time                : {:?}", proof_time);

    println!();
    println!("==========================================");
    println!("VERIFIER");
    println!("==========================================");
    println!("Selected Key              : {}", loaded.public_statement.key);
    println!("Selected KZG Witness      : {:?}", loaded.public_statement.leaf_opening_witness);
    println!("Evaluation Point          : {:?}", loaded.public_statement.evaluation_point);
    println!("Evaluation Value          : {:?}", loaded.public_statement.evaluation_value);
    println!("Polynomial Coefficients   : {:?}", loaded.public_statement.leaf_polynomial_coefficients);
    println!("Leaf Commitment           : {:?}", loaded.public_statement.leaf_commitment);
    println!("Trusted Root              : {:?}", loaded.public_statement.root_commitment);
    println!("Public SRS                : YES");
    println!("Selected Path Only        : YES");
    println!("Selected Path Levels      : {}", loaded.public_statement.path.len());
    println!("Public BabyBear Elements  : {}", loaded.public_values.len());
    println!("STARK Proof Received      : YES");
    println!("STARK Proof Count         : 1");

    println!();
    println!("==========================================");
    println!("STARK VERIFICATION");
    println!("==========================================");
    println!("Verifying STARK Proof...");

    let verification_start =
        Instant::now();

    verify(
        &config,
        &air,
        &proof,
        &loaded.public_values
    )
    .expect(
        "STARK verification failed"
    );

    let verification_time =
        verification_start.elapsed();

    println!("STARK Verification        : PASS");
    println!("Verification Time         : {:?}", verification_time);
}
// ---------------------------------------------------------
// Print Prover / Verifier Boundary
// ---------------------------------------------------------

pub fn printProverVerifierBoundary(
    loaded: &LoadedMembership
) {
    let statement = &loaded.public_statement;
    let witness = &loaded.private_witness;
    let srs = &loaded.public_parameters.srs;

    println!();
    println!("==========================================");
    println!("PROVER");
    println!("==========================================");
    println!("Selected Key              : {}", statement.key);
    println!("Selected Leaf Node        : {}", statement.leaf_node_index);
    println!("Selected Leaf Commitment  : {:?}", statement.leaf_commitment);
    println!("Selected KZG Witness      : {:?}", statement.leaf_opening_witness);
    println!("Evaluation Point          : {:?}", statement.evaluation_point);
    println!("Evaluation Value          : {:?}", statement.evaluation_value);
    println!(
        "Polynomial Coefficients   : {:?}",
        statement.leaf_polynomial_coefficients
    );
    println!("Polynomial Coeff Count    : {}", statement.leaf_polynomial_coefficients.len());
    println!("Public SRS                : YES");
    println!("SRS G1 Powers             : {}", srs.g1_powers.len());
    println!("SRS Maximum Degree        : {}", srs.max_degree);
    println!("Selected Path Levels      : {}", statement.path.len());
    println!("Private y_low             : {:?}", witness.y_low);
    println!("Private y_high            : {:?}", witness.y_high);
    println!("Internal Path Witnesses   : {}", witness.path.len());
    println!("Tau / Alpha Secret        : NO");

    println!();
    println!("Recursive Selected Path:");
    println!("  Leaf {}", statement.leaf_node_index);

    for step in &statement.path {
        if step.parent_commitment == statement.root_commitment {
            println!("    -> Node {} / ROOT", step.node_index);
        } else {
            println!("    -> Node {}", step.node_index);
        }
    }

    println!();
    println!("==========================================");
    println!("VERIFIER");
    println!("==========================================");
    println!("Selected Key              : {}", statement.key);
    println!("Selected Leaf Node        : {}", statement.leaf_node_index);
    println!("Selected Leaf Commitment  : {:?}", statement.leaf_commitment);
    println!("Selected KZG Witness      : {:?}", statement.leaf_opening_witness);
    println!("Evaluation Point          : {:?}", statement.evaluation_point);
    println!("Evaluation Value          : {:?}", statement.evaluation_value);
    println!(
        "Polynomial Coefficients   : {:?}",
        statement.leaf_polynomial_coefficients
    );
    println!("Polynomial Coeff Count    : {}", statement.leaf_polynomial_coefficients.len());
    println!("Public SRS                : YES");
    println!("SRS G1 Powers             : {}", srs.g1_powers.len());
    println!("SRS Maximum Degree        : {}", srs.max_degree);
    println!("Trusted Root              : {:?}", statement.root_commitment);
    println!("Selected Path Only        : YES");
    println!("Selected Path Levels      : {}", statement.path.len());
    println!("Internal Node Coefficients: NO");
    println!("Internal Node Witnesses   : NO");
    println!("Private y_low / y_high    : NO");
    println!("Private Trace             : NO");
    println!("Tau / Alpha Secret        : NO");
}
