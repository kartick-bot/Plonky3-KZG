mod leaf_kzg;
mod node;
mod verify_eval;
mod vplustree;

use verify_eval::verifyEval;
use vplustree::VplusTree;

use rand::prelude::*;
use sha2::{Digest, Sha256};

use leaf_kzg::{
    buildLeafEvaluations, buildLeafPolynomials, buildNodeCommitment, buildTreeCommitment,
    computeKzgWitnessRecursive, generateTreeSrs, groupEvaluationsByLeaf, internalRecordEvaluation,
    printAndVerifyLeafPolynomials,
};

// ---------------------------------------------------------
// Generate Random Unique Keys
// ---------------------------------------------------------

fn generateRandomKeys(keyCount: usize, universeMax: i64) -> Vec<i64> {
    if keyCount > universeMax as usize {
        panic!("Key count cannot be larger than key universe.");
    }

    let mut universe: Vec<i64> = (1..=universeMax).collect();

    let mut rng = rand::rng();

    universe.shuffle(&mut rng);

    universe[..keyCount].to_vec()
}

// ---------------------------------------------------------
// Generate SHA-256 Value For A Key
// ---------------------------------------------------------

fn generateHashValue(key: i64) -> [u8; 32] {
    let input = format!("Value{}", key);

    let digest = Sha256::digest(input.as_bytes());

    let mut hash = [0u8; 32];

    hash.copy_from_slice(&digest);

    hash
}

// ---------------------------------------------------------
// Convert SHA-256 Hash To Hex
// ---------------------------------------------------------

fn hashToHex(hash: &[u8; 32]) -> String {
    hash.iter().map(|byte| format!("{:02x}", byte)).collect()
}

// ---------------------------------------------------------
// Print Search Path
// ---------------------------------------------------------

fn printSearchPath(tree: &VplusTree, searchKey: i64) {
    println!();

    println!("==========================================");

    println!("Search Path for Key {}", searchKey);

    println!("==========================================");

    let (leafIndex, path) = tree.searchPath(searchKey);

    let (_, proofPath) = tree.buildProofPath(searchKey);

    // -----------------------------------------------------
    // Print Search Path
    // -----------------------------------------------------

    for (level, step) in path.iter().enumerate() {
        println!();

        println!("Level {}", level);

        println!(
            "Current Internal Node: index={} keys={:?}",
            step.nodeIndex, step.nodeKeys
        );

        println!("Chosen Child Position: {}", step.index);

        let retrievedLeft = tree.retrieveLeft(step.nodeIndex, step.index);

        let retrievedRight = tree.retrieveRight(step.nodeIndex, step.index);

        println!("retrieveLeft()  -> {:?}", retrievedLeft);

        println!("retrieveRight() -> {:?}", retrievedRight);

        match (step.leftChildIndex, &step.leftChildKeys) {
            (Some(nodeIndex), Some(keys)) => {
                println!("Left Sibling: index={} keys={:?}", nodeIndex, keys);
            }

            _ => {
                println!("Left Sibling: None");
            }
        }

        println!(
            "Chosen Child: index={} keys={:?}",
            step.chosenChildIndex, step.chosenChildKeys
        );

        match (step.rightChildIndex, &step.rightChildKeys) {
            (Some(nodeIndex), Some(keys)) => {
                println!("Right Sibling: index={} keys={:?}", nodeIndex, keys);
            }

            _ => {
                println!("Right Sibling: None");
            }
        }

        println!("          |");

        println!("          v");
    }

    // -----------------------------------------------------
    // Print Final Leaf
    // -----------------------------------------------------

    let leaf = &tree.nodes[leafIndex];

    println!("Leaf Reached: index={} keys={:?}", leafIndex, leaf.keys);

    // -----------------------------------------------------
    // Check Whether Key Exists
    // -----------------------------------------------------

    let mut keyPosition = None;

    for (index, key) in leaf.keys.iter().enumerate() {
        if *key == searchKey {
            keyPosition = Some(index);

            break;
        }
    }

    match keyPosition {
        Some(index) => {
            println!("Key {} FOUND in leaf at position {}.", searchKey, index);

            println!("SHA-256 Value: {}", hashToHex(&leaf.values[index]));
        }

        None => {
            println!("Key {} NOT FOUND in leaf.", searchKey);
        }
    }

    // -----------------------------------------------------
    // Print Proof Path
    // -----------------------------------------------------

    println!();

    println!("==========================================");

    println!("Proof Path");

    println!("==========================================");

    for (level, step) in proofPath.iter().enumerate() {
        println!();

        println!("Level {}", level);

        println!("Parent Node Index: {}", step.parentIndex);

        match (step.leftIndex, &step.leftKeys) {
            (Some(nodeIndex), Some(keys)) => {
                println!("Left:   index={} keys={:?}", nodeIndex, keys);
            }

            _ => {
                println!("Left:   None");
            }
        }

        println!(
            "Chosen: index={} keys={:?}",
            step.chosenChildIndex, step.chosenKeys
        );

        match (step.rightIndex, &step.rightKeys) {
            (Some(nodeIndex), Some(keys)) => {
                println!("Right:  index={} keys={:?}", nodeIndex, keys);
            }

            _ => {
                println!("Right:  None");
            }
        }
    }
}

// =========================================================
// Reusable Membership Verification
// =========================================================

struct MembershipVerification {
    leaf_index: usize,

    global_index: usize,

    witnesses: Vec<ark_bn254::G1Affine>,

    path: Vec<vplustree::SearchPathStep>,

    openings: Vec<VerifyEvalRecord>,
}

// ---------------------------------------------------------
// Public Inputs And Result Of One verifyEval() Call
//
// openings[0] is the leaf opening; openings[k] is the
// internal-node opening that uses witnesses[k].
// ---------------------------------------------------------

struct VerifyEvalRecord {
    evaluation_point: ark_bn254::Fr,

    evaluation_value: ark_bn254::Fr,

    commitment: ark_bn254::G1Affine,

    witness: ark_bn254::G1Affine,

    verification: bool,

    leaf_record: Option<LeafRecord>,

    public_pointers: Option<PublicPointers>,
}

// ---------------------------------------------------------
// Leaf Record Behind The Leaf Evaluation Value
//
// evaluation_value = y_low + y_high, where
// (y_high, y_low) are the two 128-bit halves of
// digest = hashKeyValue(key, value).
// ---------------------------------------------------------

struct LeafRecord {
    key: i64,

    position_in_leaf: usize,

    global_index: usize,

    value: [u8; 32],

    digest: [u8; 32],

    y_low: ark_bn254::Fr,

    y_high: ark_bn254::Fr,
}

// ---------------------------------------------------------
// Internal Record Behind The Internal Evaluation Value
//
// evaluation_value = internalRecordEvaluation(key, LPC, RPC)
// ---------------------------------------------------------

struct PublicPointers {
    key: i64,

    lpc_child_node: usize,

    lpc: ark_bn254::G1Affine,

    rpc_child_node: usize,

    rpc: ark_bn254::G1Affine,

    chosen_mapping: &'static str,
}

// ---------------------------------------------------------
// Verify One Complete Membership Proof
//
// Verifies:
//
//   1. leaf KZG opening at the global leaf-level index,
//   2. every internal-node KZG opening,
//   3. child-to-parent commitment linkage,
//   4. termination at the trusted root commitment.
//
// This is generic over tree order and tree depth.
// ---------------------------------------------------------

fn verifyMembershipPath(
    tree: &VplusTree,
    key: i64,
    evaluations: &[leaf_kzg::LeafEvaluation],
    polynomials: &[leaf_kzg::LeafPolynomial],
    srs: &leaf_kzg::KzgSrs,
    rootCommitment: &ark_bn254::G1Affine,
    label: &str,
) -> MembershipVerification {
    let witnesses = computeKzgWitnessRecursive(tree, tree.root, key, polynomials, srs);

    let (leafIndex, path) = tree.searchPath(key);

    assert_eq!(
        witnesses.len(),
        path.len() + 1,
        "{} witness count does not match leaf-to-root path",
        label
    );

    let leafEvaluation = evaluations
        .iter()
        .find(|evaluation| evaluation.leaf_node_index == leafIndex && evaluation.key == key)
        .expect("Membership key evaluation/global index not found");

    let globalIndex = leafEvaluation.global_index;

    let leafPolynomial = polynomials
        .iter()
        .find(|polynomial| polynomial.leaf_node_index == leafIndex)
        .expect("Membership leaf polynomial not found");

    let leafEvaluationPoint = ark_bn254::Fr::from(globalIndex as u64);

    let leafEvaluationValue =
        leaf_kzg::evaluatePolynomial(&leafPolynomial.coefficients, leafEvaluationPoint);

    let leafVerification = verifyEval(
        &leafPolynomial.commitment,
        leafEvaluationPoint,
        leafEvaluationValue,
        &witnesses[0],
        srs,
    );

    let mut openings: Vec<VerifyEvalRecord> = Vec::new();

    openings.push(VerifyEvalRecord {
        evaluation_point: leafEvaluationPoint,

        evaluation_value: leafEvaluationValue,

        commitment: leafPolynomial.commitment,

        witness: witnesses[0],

        verification: leafVerification,

        leaf_record: Some(LeafRecord {
            key: leafEvaluation.key,

            position_in_leaf: leafEvaluation.position_in_leaf,

            global_index: leafEvaluation.global_index,

            value: tree.nodes[leafIndex].values[leafEvaluation.position_in_leaf],

            digest: leafEvaluation.digest,

            y_low: leafEvaluation.y_low,

            y_high: leafEvaluation.y_high,
        }),

        public_pointers: None,
    });

    println!();

    println!("==========================================");

    println!("Membership Proof: {}", label);

    println!("==========================================");

    println!("Key          : {}", key);

    println!("Global Index : {}", globalIndex);

    println!("Leaf Node    : {}", leafIndex);

    println!("Witness Count: {}", witnesses.len());

    println!();

    println!("Witness 0");

    println!("  Node Type  : Leaf");

    println!("  Node Index : {}", leafIndex);

    println!("  x          : {}", globalIndex);

    println!("  Witness    : {:?}", witnesses[0]);

    println!("  verifyEval : {}", leafVerification);

    assert!(leafVerification, "{} leaf KZG verification failed", label);

    let mut verifiedChildCommitment = leafPolynomial.commitment;

    for (level, step) in path.iter().rev().enumerate() {
        let witnessIndex = level + 1;

        let nodeIndex = step.nodeIndex;

        let node = &tree.nodes[nodeIndex];

        let (recordPosition, chosenSide) = if step.index < node.keys.len() {
            (step.index, "LPC")
        } else {
            (node.keys.len() - 1, "RPC")
        };

        let leftChildIndex = node.children[recordPosition];

        let rightChildIndex = node.children[recordPosition + 1];

        let leftCommitment = buildNodeCommitment(tree, leftChildIndex, polynomials, srs);

        let rightCommitment = buildNodeCommitment(tree, rightChildIndex, polynomials, srs);

        let authenticatedChildCommitment = if step.index < node.keys.len() {
            assert_eq!(
                step.chosenChildIndex, leftChildIndex,
                "{} chosen child does not match LPC child at node {}",
                label, nodeIndex
            );

            leftCommitment
        } else {
            assert_eq!(
                step.chosenChildIndex, rightChildIndex,
                "{} chosen child does not match RPC child at node {}",
                label, nodeIndex
            );

            rightCommitment
        };

        assert_eq!(
            authenticatedChildCommitment, verifiedChildCommitment,
            "{} commitment linkage failed at witness {} node {}",
            label, witnessIndex, nodeIndex
        );

        let internalEvaluationValue =
            internalRecordEvaluation(node.keys[recordPosition], &leftCommitment, &rightCommitment);

        let internalEvaluationPoint = ark_bn254::Fr::from(recordPosition as u64);

        let nodeCommitment = buildNodeCommitment(tree, nodeIndex, polynomials, srs);

        let verification = verifyEval(
            &nodeCommitment,
            internalEvaluationPoint,
            internalEvaluationValue,
            &witnesses[witnessIndex],
            srs,
        );

        openings.push(VerifyEvalRecord {
            evaluation_point: internalEvaluationPoint,

            evaluation_value: internalEvaluationValue,

            commitment: nodeCommitment,

            witness: witnesses[witnessIndex],

            verification,

            leaf_record: None,

            public_pointers: Some(PublicPointers {
                key: node.keys[recordPosition],

                lpc_child_node: leftChildIndex,

                lpc: leftCommitment,

                rpc_child_node: rightChildIndex,

                rpc: rightCommitment,

                chosen_mapping: chosenSide,
            }),
        });

        println!();

        println!("Witness {}", witnessIndex);

        println!("  Node Type  : Internal");

        println!("  Node Index : {}", nodeIndex);

        println!("  Record     : {}", recordPosition);

        println!("  Mapping    : {}", chosenSide);

        println!("  Child Node : {}", step.chosenChildIndex);

        println!("  Witness    : {:?}", witnesses[witnessIndex]);

        println!("  verifyEval : {}", verification);

        assert!(
            verification,
            "{} internal-node KZG verification failed at witness {} node {}",
            label, witnessIndex, nodeIndex
        );

        verifiedChildCommitment = nodeCommitment;
    }

    assert_eq!(
        verifiedChildCommitment, *rootCommitment,
        "{} proof does not terminate at trusted root commitment",
        label
    );

    println!();

    println!("Trusted Root Check : PASS");

    println!("{} Membership Proof: PASS", label);

    println!("==========================================");

    MembershipVerification {
        leaf_index: leafIndex,

        global_index: globalIndex,

        witnesses,

        path,

        openings,
    }
}

// ---------------------------------------------------------
// Build and Test One Tree
// ---------------------------------------------------------

fn buildTree(order: usize, keys: &Vec<i64>) {
    println!();

    println!("==========================================");

    println!("V+ Tree Order: {}", order);

    println!("Number of Keys: {}", keys.len());

    println!("==========================================");

    let mut tree = VplusTree::new(order);

    // -----------------------------------------------------
    // Insert Key-Value Pairs
    // -----------------------------------------------------

    for key in keys {
        let value = generateHashValue(*key);

        tree.insert(*key, value);
    }

    // -----------------------------------------------------
    // Print Tree
    // -----------------------------------------------------

    println!();

    println!("Tree Structure:");

    tree.show();

    // -----------------------------------------------------
    // KZG SRS
    // -----------------------------------------------------

    let max_degree = order - 2;

    let srs = generateTreeSrs(max_degree);

    println!();

    println!("==========================================");

    println!("KZG SRS");

    println!("==========================================");

    println!("Tree Order: {}", order);

    println!("Maximum Polynomial Degree: {}", srs.max_degree);

    println!("Number of G1 Powers: {}", srs.g1_powers.len());

    println!("G2 Generator Present: true");

    println!("Tau*G2 Present: true");

    println!();

    println!("Height: {}", tree.getHeight());

    println!("Node Count: {}", tree.countNodes());

    println!("Key Count: {}", tree.countKeys());

    println!();

    println!("Basic Validation: {}", tree.validate());

    println!();

    println!("Full Verification:");

    tree.verifyTree();

    // -----------------------------------------------------
    // Build Leaf-Level Evaluations
    // -----------------------------------------------------

    let evaluations = buildLeafEvaluations(&tree);

    // -----------------------------------------------------
    // Group Evaluations By Leaf
    // -----------------------------------------------------

    let grouped = groupEvaluationsByLeaf(&evaluations);

    // -----------------------------------------------------
    // Build Leaf Polynomials
    // -----------------------------------------------------

    let polynomials = buildLeafPolynomials(&grouped, &srs);

    // -----------------------------------------------------
    // Verify Leaf Polynomials
    // -----------------------------------------------------

    printAndVerifyLeafPolynomials(&grouped, &polynomials);

    // -----------------------------------------------------
    // Build Root Commitment
    // -----------------------------------------------------

    let rootCommitment = buildTreeCommitment(&tree, &polynomials, &srs);

    // -----------------------------------------------------
    // Choose Search Key
    // -----------------------------------------------------

    // The smallest key has no immediate predecessor, so
    // take the first random key that is not the smallest.

    let smallestKey = *keys.iter().min().expect("Key list is empty");

    let searchKey = keys
        .iter()
        .copied()
        .find(|key| *key != smallestKey)
        .expect("Need at least two keys for a predecessor proof");

    // =====================================================
    // MEMBERSHIP PROOF 1: QUERIED KEY
    // =====================================================

    let requestedProof = verifyMembershipPath(
        &tree,
        searchKey,
        &evaluations,
        &polynomials,
        &srs,
        &rootCommitment,
        "Queried Key",
    );

    let leafIndex = requestedProof.leaf_index;

    let requestedGlobalIndex = requestedProof.global_index;

    let witnesses = requestedProof.witnesses;

    let path = requestedProof.path;

    let openings = requestedProof.openings;

    // =====================================================
    // MEMBERSHIP PROOF 2: IMMEDIATE PREDECESSOR
    // =====================================================

    assert!(
        requestedGlobalIndex > 0,
        "Requested key is the first key in the tree and has no predecessor"
    );

    let predecessorGlobalIndex = requestedGlobalIndex - 1;

    let predecessorEvaluation = evaluations
        .iter()
        .find(|evaluation| evaluation.global_index == predecessorGlobalIndex)
        .expect("Immediate predecessor evaluation not found");

    let predecessorKey = predecessorEvaluation.key;

    let predecessorProof = verifyMembershipPath(
        &tree,
        predecessorKey,
        &evaluations,
        &polynomials,
        &srs,
        &rootCommitment,
        "Immediate Predecessor",
    );

    assert_eq!(
        predecessorProof.global_index, predecessorGlobalIndex,
        "Verified predecessor global index does not match expected immediate predecessor index"
    );

    // =====================================================
    // ORDERING CHECKS
    // =====================================================

    println!();

    println!("==========================================");

    println!("Ordering Checks");

    println!("==========================================");

    println!("Queried Key              : {}", searchKey);

    println!("Queried Global Index     : {}", requestedGlobalIndex);

    println!("Immediate Predecessor    : {}", predecessorKey);

    println!(
        "Predecessor Global Index : {}",
        predecessorProof.global_index
    );

    assert_eq!(
        requestedGlobalIndex,
        predecessorProof.global_index + 1,
        "Immediate-predecessor index adjacency check failed"
    );

    println!(
        "Index adjacency check: {} + 1 = {} : PASS",
        predecessorProof.global_index, requestedGlobalIndex
    );

    assert!(
        predecessorKey < searchKey,
        "Immediate-predecessor key ordering check failed"
    );

    println!(
        "Key ordering check: {} < {} : PASS",
        predecessorKey, searchKey
    );

    println!("Immediate predecessor ordering proof: PASS");

    println!("==========================================");

    // =====================================================
    // Requested-Key Diagnostic Information
    // =====================================================

    let expectedWitnessCount = path.len() + 1;

    assert_eq!(
        witnesses.len(),
        expectedWitnessCount,
        "Witness count does not match leaf-to-root path"
    );

    if !path.is_empty() {
        assert_eq!(
            path[0].nodeIndex, tree.root,
            "Witness path does not start at root"
        );

        assert_eq!(
            path[path.len() - 1].chosenChildIndex,
            leafIndex,
            "Witness path does not end at expected leaf"
        );
    }

    println!();

    println!("==========================================");

    println!("KZG Witness And Commitment Check");

    println!("==========================================");

    println!("Search Key: {}", searchKey);

    println!("Witness Count: {}", witnesses.len());

    println!("Expected Witness Count: {}", expectedWitnessCount);

    println!("Witness Count Check: PASS");

    let leafCommitment = polynomials
        .iter()
        .find(|polynomial| polynomial.leaf_node_index == leafIndex)
        .expect("Leaf commitment not found")
        .commitment;

    println!();

    println!("Witness 0");

    println!("  Node Type   : Leaf");

    println!("  Node Index  : {}", leafIndex);

    println!("  Commitment  : {:?}", leafCommitment);

    println!("  Witness     : {:?}", witnesses[0]);

    println!("  Mapping     : PASS");

    println!("  Math Check  : PASS");

    for (level, step) in path.iter().rev().enumerate() {
        let witnessIndex = level + 1;

        let recordPosition = if step.index < step.nodeKeys.len() {
            step.index
        } else {
            step.nodeKeys.len() - 1
        };

        let mappingSide = if step.index < step.nodeKeys.len() {
            "LPC"
        } else {
            "RPC"
        };

        println!();

        println!("Witness {}", witnessIndex);

        println!("  Node Type   : Internal");

        println!("  Node Index  : {}", step.nodeIndex);

        println!("  Record      : {}", recordPosition);

        println!("  Mapping     : {}", mappingSide);

        println!("  Child Node  : {}", step.chosenChildIndex);

        println!("  Witness     : {:?}", witnesses[witnessIndex]);

        if step.nodeIndex == tree.root {
            println!("  Commitment  : {:?}", rootCommitment);
        } else {
            println!("  Commitment  : verified by recursive commitment check");
        }

        println!("  Mapping     : PASS");

        println!("  Math Check  : PASS");
    }

    println!();

    println!("==========================================");

    println!("KZG Verification Summary");

    println!("==========================================");

    println!("Polynomial Division       : PASS");

    println!("Division Remainder        : PASS");

    println!("Polynomial Reconstruction : PASS");

    println!("Commitment Recalculation  : PASS");

    println!("Leaf To Root Mapping      : PASS");

    println!("Witness Generation        : PASS");

    // -----------------------------------------------------
    // Write JSON Files
    // -----------------------------------------------------

    writeJsonFiles(
        &mut tree,
        order,
        &polynomials,
        &srs,
        searchKey,
        &witnesses,
        &rootCommitment,
        &openings,
        &predecessorProof,
        predecessorKey,
        requestedGlobalIndex,
        &evaluations,
    );

    println!("Membership ok");
}

// ---------------------------------------------------------
// Main
// ---------------------------------------------------------

fn main() {
    let orders = vec![5, 6];

    let power: u32 = 3;

    let universePowers = 16..=16u32;

    for order in orders {
        let keyBase: usize = order;

        let keyCount = keyBase.pow(power);

        for universePower in universePowers.clone() {
            let universeMax = 2_i64.pow(universePower).max((keyCount as i64) * 2);

            println!("==========================================");

            println!("Experiment Configuration");

            println!("==========================================");

            println!("Tree Order: {}", order);

            println!("Keys: {}^{} = {}", keyBase, power, keyCount);

            println!(
                "Key Universe: 1 to 2^{} = 1 to {}",
                universePower, universeMax
            );

            let keys = generateRandomKeys(keyCount, universeMax);

            buildTree(order, &keys);

            // ---------------------------------------------
            // Keep This Run's JSON Files
            // ---------------------------------------------

            for fileName in ["vplus_tree", "kzg_details"] {
                let runFileName = format!(
                    "{}_keys{}pow{}_universe2pow{}.json",
                    fileName, keyBase, power, universePower
                );

                std::fs::rename(format!("{}.json", fileName), &runFileName)
                    .expect("Could not rename JSON file");

                println!("Saved: {}", runFileName);
            }
        }
    }
}

// ---------------------------------------------------------
// Write Tree And KZG Data To JSON Files
// ---------------------------------------------------------

fn writeJsonFiles(
    tree: &mut VplusTree,
    order: usize,
    polynomials: &[leaf_kzg::LeafPolynomial],
    srs: &leaf_kzg::KzgSrs,
    searchKey: i64,
    witnesses: &[ark_bn254::G1Affine],
    rootCommitment: &ark_bn254::G1Affine,
    openings: &[VerifyEvalRecord],
    predecessorProof: &MembershipVerification,
    predecessorKey: i64,
    searchGlobalIndex: usize,
    evaluations: &[leaf_kzg::LeafEvaluation],
) {
    use std::io::Write;

    let jsonString = |value: String| {
        value
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t")
    };

    // -----------------------------------------------------
    // Write public_pointers / verify_eval Blocks
    // -----------------------------------------------------

    let writePublicPointers = |file: &mut std::fs::File, pointers: &PublicPointers| {
        writeln!(file, "      \"public_pointers\": {{").unwrap();

        writeln!(file, "        \"key\": {},", pointers.key).unwrap();

        writeln!(
            file,
            "        \"LPC_child_node\": {},",
            pointers.lpc_child_node
        )
        .unwrap();

        writeln!(
            file,
            "        \"LPC\": \"{}\",",
            jsonString(format!("{:?}", pointers.lpc))
        )
        .unwrap();

        writeln!(
            file,
            "        \"RPC_child_node\": {},",
            pointers.rpc_child_node
        )
        .unwrap();

        writeln!(
            file,
            "        \"RPC\": \"{}\",",
            jsonString(format!("{:?}", pointers.rpc))
        )
        .unwrap();

        writeln!(
            file,
            "        \"chosen_mapping\": \"{}\"",
            pointers.chosen_mapping
        )
        .unwrap();

        writeln!(file, "      }},").unwrap();
    };

    let writeVerifyEval = |file: &mut std::fs::File, opening: &VerifyEvalRecord| {
        writeln!(file, "      \"verify_eval\": {{").unwrap();

        writeln!(
            file,
            "        \"evaluation_point\": \"{}\",",
            opening.evaluation_point
        )
        .unwrap();

        writeln!(
            file,
            "        \"evaluation_value\": \"{}\",",
            opening.evaluation_value
        )
        .unwrap();

        writeln!(
            file,
            "        \"commitment\": \"{}\",",
            jsonString(format!("{:?}", opening.commitment))
        )
        .unwrap();

        writeln!(
            file,
            "        \"witness\": \"{}\",",
            jsonString(format!("{:?}", opening.witness))
        )
        .unwrap();

        writeln!(file, "        \"verification\": {}", opening.verification).unwrap();

        writeln!(file, "      }}").unwrap();
    };

    let writeLeafRecord = |file: &mut std::fs::File, record: &LeafRecord| {
        writeln!(file, "      \"leaf_record\": {{").unwrap();

        writeln!(file, "        \"key\": {},", record.key).unwrap();

        writeln!(
            file,
            "        \"position_in_leaf\": {},",
            record.position_in_leaf
        )
        .unwrap();

        writeln!(file, "        \"global_index\": {},", record.global_index).unwrap();

        writeln!(
            file,
            "        \"value_sha256\": \"{}\",",
            hashToHex(&record.value)
        )
        .unwrap();

        writeln!(
            file,
            "        \"key_value_digest\": \"{}\",",
            hashToHex(&record.digest)
        )
        .unwrap();

        writeln!(file, "        \"y_low\": \"{}\",", record.y_low).unwrap();

        writeln!(file, "        \"y_high\": \"{}\"", record.y_high).unwrap();

        writeln!(file, "      }},").unwrap();
    };

    // -----------------------------------------------------
    // Build Commitment Table
    // -----------------------------------------------------

    let originalRoot = tree.root;

    let mut nodeCommitments: Vec<Option<ark_bn254::G1Affine>> = vec![None; tree.nodes.len()];

    for nodeIndex in 0..tree.nodes.len() {
        tree.root = nodeIndex;

        nodeCommitments[nodeIndex] = Some(buildTreeCommitment(tree, polynomials, srs));
    }

    tree.root = originalRoot;

    assert_eq!(
        nodeCommitments[tree.root]
            .as_ref()
            .expect("Root commitment missing"),
        rootCommitment,
        "Root commitment does not match existing root commitment"
    );

    // -----------------------------------------------------
    // Search Path
    // -----------------------------------------------------

    let (leafIndex, path) = tree.searchPath(searchKey);

    let expectedWitnessCount = path.len() + 1;

    assert_eq!(
        witnesses.len(),
        expectedWitnessCount,
        "Witness count does not match leaf-to-root path"
    );

    assert_eq!(
        openings.len(),
        witnesses.len(),
        "verifyEval record count does not match witness count"
    );

    assert_eq!(
        predecessorProof.openings.len(),
        predecessorProof.witnesses.len(),
        "Predecessor verifyEval record count does not match witness count"
    );

    // -----------------------------------------------------
    // KZG Opening Of Every Internal-Node Record
    //
    // Rebuilds each internal polynomial from the same
    // points used by buildNodeCommitment():
    //
    //     (i, internalRecordEvaluation(key_i, LPC_i, RPC_i))
    //
    // and opens it at every record position i.
    //
    // internalOpenings[node][i] =
    //     (evaluation_value, witness, verifyEval result)
    // -----------------------------------------------------

    let mut internalOpenings: Vec<Vec<(ark_bn254::Fr, ark_bn254::G1Affine, bool)>> =
        vec![Vec::new(); tree.nodes.len()];

    for nodeIndex in 0..tree.nodes.len() {
        let node = &tree.nodes[nodeIndex];

        if node.is_leaf {
            continue;
        }

        let nodeCommitment = nodeCommitments[nodeIndex].expect("Internal commitment missing");

        let mut points = Vec::new();

        for i in 0..node.keys.len() {
            let y = internalRecordEvaluation(
                node.keys[i],
                &nodeCommitments[node.children[i]].expect("LPC commitment missing"),
                &nodeCommitments[node.children[i + 1]].expect("RPC commitment missing"),
            );

            points.push((ark_bn254::Fr::from(i as u64), y));
        }

        let coefficients = leaf_kzg::lagrangeInterpolate(&points);

        assert_eq!(
            leaf_kzg::commitPolynomial(&coefficients, srs),
            nodeCommitment,
            "Internal polynomial does not match commitment at node {}",
            nodeIndex
        );

        for (x, y) in &points {
            let witness = leaf_kzg::computeKzgWitness(&coefficients, *x, srs);

            let verification = verifyEval(&nodeCommitment, *x, *y, &witness, srs);

            assert!(
                verification,
                "Internal opening failed at node {}",
                nodeIndex
            );

            internalOpenings[nodeIndex].push((*y, witness, verification));
        }
    }

    // The search-path witnesses must be the same openings.

    for (level, step) in path.iter().rev().enumerate() {
        let recordPosition = if step.index < step.nodeKeys.len() {
            step.index
        } else {
            step.nodeKeys.len() - 1
        };

        assert_eq!(
            internalOpenings[step.nodeIndex][recordPosition].1,
            witnesses[level + 1],
            "Search-path witness mismatch at node {}",
            step.nodeIndex
        );
    }

    // =====================================================
    // FILE 1: vplus_tree.json
    // =====================================================

    let mut treeFile =
        std::fs::File::create("vplus_tree.json").expect("Could not create vplus_tree.json");

    let mut asciiTree = String::new();

    let mut stack: Vec<(usize, String, bool, bool)> = vec![(tree.root, String::new(), true, true)];

    while let Some((nodeIndex, prefix, isLast, isRoot)) = stack.pop() {
        let node = &tree.nodes[nodeIndex];

        if isRoot {
            asciiTree.push_str(&format!("ROOT [{}] keys={:?}\n", nodeIndex, node.keys));
        } else {
            let branch = "+-- ";

            let nodeType = if node.is_leaf { "LEAF" } else { "PARENT" };

            asciiTree.push_str(&format!(
                "{}{}{} [{}] keys={:?}\n",
                prefix, branch, nodeType, nodeIndex, node.keys
            ));
        }

        if !node.is_leaf {
            let childPrefix = if isRoot {
                String::new()
            } else {
                format!("{}{}", prefix, if isLast { "    " } else { "|   " })
            };

            for i in (0..node.children.len()).rev() {
                stack.push((
                    node.children[i],
                    childPrefix.clone(),
                    i == node.children.len() - 1,
                    false,
                ));
            }
        }
    }

    let asciiTree = jsonString(asciiTree);

    writeln!(treeFile, "{{").unwrap();

    writeln!(treeFile, "  \"order\": {},", tree.order).unwrap();

    writeln!(treeFile, "  \"height\": {},", tree.getHeight()).unwrap();

    writeln!(treeFile, "  \"number_of_keys\": {},", tree.countKeys()).unwrap();

    writeln!(treeFile, "  \"number_of_nodes\": {},", tree.countNodes()).unwrap();

    writeln!(treeFile, "  \"root_node\": {},", tree.root).unwrap();

    writeln!(treeFile, "  \"ascii_tree\": \"{}\",", asciiTree).unwrap();

    writeln!(treeFile, "  \"nodes\": [").unwrap();

    for (nodeIndex, node) in tree.nodes.iter().enumerate() {
        let nodeType = if nodeIndex == tree.root {
            "root"
        } else if node.is_leaf {
            "leaf"
        } else {
            "parent"
        };

        writeln!(treeFile, "    {{").unwrap();

        writeln!(treeFile, "      \"node_index\": {},", nodeIndex).unwrap();

        writeln!(treeFile, "      \"node_type\": \"{}\",", nodeType).unwrap();

        match node.parent_node {
            Some(parentIndex) => {
                writeln!(treeFile, "      \"parent_node\": {},", parentIndex).unwrap();
            }

            None => {
                writeln!(treeFile, "      \"parent_node\": null,").unwrap();
            }
        }

        writeln!(treeFile, "      \"keys\": {:?},", node.keys).unwrap();

        writeln!(treeFile, "      \"children\": {:?},", node.children).unwrap();

        writeln!(treeFile, "      \"is_leaf\": {}", node.is_leaf).unwrap();

        if nodeIndex + 1 < tree.nodes.len() {
            writeln!(treeFile, "    }},").unwrap();
        } else {
            writeln!(treeFile, "    }}").unwrap();
        }
    }

    writeln!(treeFile, "  ]").unwrap();

    writeln!(treeFile, "}}").unwrap();

    // =====================================================
    // FILE 2: kzg_details.json
    // =====================================================

    let mut kzgFile =
        std::fs::File::create("kzg_details.json").expect("Could not create kzg_details.json");

    let baseFieldModulus = <ark_bn254::Fq as ark_ff::PrimeField>::MODULUS.to_string();

    let scalarFieldModulus = <ark_bn254::Fr as ark_ff::PrimeField>::MODULUS.to_string();

    writeln!(kzgFile, "{{").unwrap();

    // -----------------------------------------------------
    // Experiment
    // -----------------------------------------------------

    writeln!(kzgFile, "  \"experiment\": {{").unwrap();

    writeln!(kzgFile, "    \"order\": {},", order).unwrap();

    writeln!(kzgFile, "    \"height\": {},", tree.getHeight()).unwrap();

    writeln!(kzgFile, "    \"number_of_keys\": {},", tree.countKeys()).unwrap();

    writeln!(kzgFile, "    \"number_of_nodes\": {},", tree.countNodes()).unwrap();

    writeln!(kzgFile, "    \"search_key\": {},", searchKey).unwrap();

    writeln!(kzgFile, "    \"leaf_node\": {},", leafIndex).unwrap();

    writeln!(kzgFile, "    \"witness_count\": {}", witnesses.len()).unwrap();

    writeln!(kzgFile, "  }},").unwrap();

    // -----------------------------------------------------
    // Curve
    // -----------------------------------------------------

    writeln!(kzgFile, "  \"curve\": {{").unwrap();

    writeln!(kzgFile, "    \"name\": \"BN254\",").unwrap();

    writeln!(kzgFile, "    \"G1_curve_equation\": \"y^2 = x^3 + 3\",").unwrap();

    writeln!(
        kzgFile,
        "    \"base_field_modulus_p\": \"{}\",",
        baseFieldModulus
    )
    .unwrap();

    writeln!(
        kzgFile,
        "    \"scalar_field_modulus_r\": \"{}\"",
        scalarFieldModulus
    )
    .unwrap();

    writeln!(kzgFile, "  }},").unwrap();

    // -----------------------------------------------------
    // SRS
    // -----------------------------------------------------

    writeln!(kzgFile, "  \"SRS\": {{").unwrap();

    writeln!(kzgFile, "    \"max_degree\": {},", srs.max_degree).unwrap();

    match srs.g1_powers.first() {
        Some(g1) => {
            writeln!(
                kzgFile,
                "    \"G1\": \"{}\",",
                jsonString(format!("{:?}", g1))
            )
            .unwrap();
        }

        None => {
            writeln!(kzgFile, "    \"G1\": null,").unwrap();
        }
    }

    writeln!(
        kzgFile,
        "    \"G2\": \"{}\",",
        jsonString(format!("{:?}", srs.g2))
    )
    .unwrap();

    writeln!(
        kzgFile,
        "    \"tau_G2\": \"{}\",",
        jsonString(format!("{:?}", srs.tau_g2))
    )
    .unwrap();

    writeln!(kzgFile, "    \"G1_powers\": [").unwrap();

    for i in 0..srs.g1_powers.len() {
        let value = jsonString(format!("{:?}", srs.g1_powers[i]));

        if i + 1 < srs.g1_powers.len() {
            writeln!(kzgFile, "      \"{}\",", value).unwrap();
        } else {
            writeln!(kzgFile, "      \"{}\"", value).unwrap();
        }
    }

    writeln!(kzgFile, "    ]").unwrap();

    writeln!(kzgFile, "  }},").unwrap();

    // =====================================================
    // ROOT
    // =====================================================

    let root = &tree.nodes[tree.root];

    writeln!(kzgFile, "  \"root\": {{").unwrap();

    writeln!(kzgFile, "    \"node_index\": {},", tree.root).unwrap();

    writeln!(kzgFile, "    \"keys\": {:?},", root.keys).unwrap();

    writeln!(kzgFile, "    \"children\": {:?},", root.children).unwrap();

    writeln!(
        kzgFile,
        "    \"commitment\": \"{}\",",
        jsonString(format!("{:?}", rootCommitment))
    )
    .unwrap();

    writeln!(
        kzgFile,
        "    \"witness\": \"{}\",",
        jsonString(format!("{:?}", witnesses[witnesses.len() - 1]))
    )
    .unwrap();

    writeln!(kzgFile, "    \"records\": [").unwrap();

    for i in 0..root.keys.len() {
        let leftChild = root.children[i];

        let rightChild = root.children[i + 1];

        let leftCommitment = nodeCommitments[leftChild]
            .as_ref()
            .expect("LPC commitment missing");

        let rightCommitment = nodeCommitments[rightChild]
            .as_ref()
            .expect("RPC commitment missing");

        writeln!(kzgFile, "      {{").unwrap();

        writeln!(kzgFile, "        \"record_position\": {},", i).unwrap();

        writeln!(kzgFile, "        \"key\": {},", root.keys[i]).unwrap();

        writeln!(kzgFile, "        \"LPC_child_node\": {},", leftChild).unwrap();

        writeln!(
            kzgFile,
            "        \"LPC\": \"{}\",",
            jsonString(format!("{:?}", leftCommitment))
        )
        .unwrap();

        writeln!(kzgFile, "        \"RPC_child_node\": {},", rightChild).unwrap();

        writeln!(
            kzgFile,
            "        \"RPC\": \"{}\",",
            jsonString(format!("{:?}", rightCommitment))
        )
        .unwrap();

        let (evaluationValue, recordWitness, recordVerification) = internalOpenings[tree.root][i];

        writeln!(kzgFile, "        \"evaluation_point\": \"{}\",", i).unwrap();

        writeln!(
            kzgFile,
            "        \"evaluation_value\": \"{}\",",
            evaluationValue
        )
        .unwrap();

        writeln!(
            kzgFile,
            "        \"witness\": \"{}\",",
            jsonString(format!("{:?}", recordWitness))
        )
        .unwrap();

        writeln!(kzgFile, "        \"verification\": {}", recordVerification).unwrap();

        if i + 1 < root.keys.len() {
            writeln!(kzgFile, "      }},").unwrap();
        } else {
            writeln!(kzgFile, "      }}").unwrap();
        }
    }

    writeln!(kzgFile, "    ]").unwrap();

    writeln!(kzgFile, "  }},").unwrap();

    // =====================================================
    // PARENT / INTERNAL NODES
    // =====================================================

    writeln!(kzgFile, "  \"parents\": [").unwrap();

    let parentCount = tree
        .nodes
        .iter()
        .enumerate()
        .filter(|(nodeIndex, node)| *nodeIndex != tree.root && !node.is_leaf)
        .count();

    let mut parentWritten = 0usize;

    for (nodeIndex, node) in tree.nodes.iter().enumerate() {
        if nodeIndex == tree.root || node.is_leaf {
            continue;
        }

        let commitment = nodeCommitments[nodeIndex]
            .as_ref()
            .expect("Parent commitment missing");

        let mut witnessIndex: Option<usize> = None;

        let mut openingPosition: Option<usize> = None;

        let mut mappingSide: Option<&str> = None;

        let mut chosenChild: Option<usize> = None;

        for (level, step) in path.iter().rev().enumerate() {
            if step.nodeIndex == nodeIndex {
                witnessIndex = Some(level + 1);

                if step.index < step.nodeKeys.len() {
                    openingPosition = Some(step.index);

                    mappingSide = Some("LPC");
                } else {
                    openingPosition = Some(step.nodeKeys.len() - 1);

                    mappingSide = Some("RPC");
                }

                chosenChild = Some(step.chosenChildIndex);

                break;
            }
        }

        writeln!(kzgFile, "    {{").unwrap();

        writeln!(kzgFile, "      \"node_index\": {},", nodeIndex).unwrap();

        match node.parent_node {
            Some(parentIndex) => {
                writeln!(kzgFile, "      \"parent_node\": {},", parentIndex).unwrap();
            }

            None => {
                writeln!(kzgFile, "      \"parent_node\": null,").unwrap();
            }
        }

        writeln!(kzgFile, "      \"keys\": {:?},", node.keys).unwrap();

        writeln!(kzgFile, "      \"children\": {:?},", node.children).unwrap();

        writeln!(
            kzgFile,
            "      \"commitment\": \"{}\",",
            jsonString(format!("{:?}", commitment))
        )
        .unwrap();

        match witnessIndex {
            Some(index) => {
                writeln!(
                    kzgFile,
                    "      \"witness\": \"{}\",",
                    jsonString(format!("{:?}", witnesses[index]))
                )
                .unwrap();

                writeln!(
                    kzgFile,
                    "      \"opening_position\": {},",
                    openingPosition.unwrap()
                )
                .unwrap();

                writeln!(kzgFile, "      \"mapping\": \"{}\",", mappingSide.unwrap()).unwrap();

                writeln!(kzgFile, "      \"chosen_child\": {},", chosenChild.unwrap()).unwrap();
            }

            None => {
                writeln!(kzgFile, "      \"witness\": null,").unwrap();

                writeln!(kzgFile, "      \"opening_position\": null,").unwrap();

                writeln!(kzgFile, "      \"mapping\": null,").unwrap();

                writeln!(kzgFile, "      \"chosen_child\": null,").unwrap();
            }
        }

        writeln!(kzgFile, "      \"records\": [").unwrap();

        for i in 0..node.keys.len() {
            let leftChild = node.children[i];

            let rightChild = node.children[i + 1];

            let leftCommitment = nodeCommitments[leftChild]
                .as_ref()
                .expect("LPC commitment missing");

            let rightCommitment = nodeCommitments[rightChild]
                .as_ref()
                .expect("RPC commitment missing");

            writeln!(kzgFile, "        {{").unwrap();

            writeln!(kzgFile, "          \"record_position\": {},", i).unwrap();

            writeln!(kzgFile, "          \"key\": {},", node.keys[i]).unwrap();

            writeln!(kzgFile, "          \"LPC_child_node\": {},", leftChild).unwrap();

            writeln!(
                kzgFile,
                "          \"LPC\": \"{}\",",
                jsonString(format!("{:?}", leftCommitment))
            )
            .unwrap();

            writeln!(kzgFile, "          \"RPC_child_node\": {},", rightChild).unwrap();

            writeln!(
                kzgFile,
                "          \"RPC\": \"{}\",",
                jsonString(format!("{:?}", rightCommitment))
            )
            .unwrap();

            let (evaluationValue, recordWitness, recordVerification) =
                internalOpenings[nodeIndex][i];

            writeln!(kzgFile, "          \"evaluation_point\": \"{}\",", i).unwrap();

            writeln!(
                kzgFile,
                "          \"evaluation_value\": \"{}\",",
                evaluationValue
            )
            .unwrap();

            writeln!(
                kzgFile,
                "          \"witness\": \"{}\",",
                jsonString(format!("{:?}", recordWitness))
            )
            .unwrap();

            writeln!(
                kzgFile,
                "          \"verification\": {}",
                recordVerification
            )
            .unwrap();

            if i + 1 < node.keys.len() {
                writeln!(kzgFile, "        }},").unwrap();
            } else {
                writeln!(kzgFile, "        }}").unwrap();
            }
        }

        writeln!(kzgFile, "      ]").unwrap();

        parentWritten += 1;

        if parentWritten < parentCount {
            writeln!(kzgFile, "    }},").unwrap();
        } else {
            writeln!(kzgFile, "    }}").unwrap();
        }
    }

    writeln!(kzgFile, "  ],").unwrap();

    // =====================================================
    // LEAF NODES
    // =====================================================

    writeln!(kzgFile, "  \"leaves\": [").unwrap();

    let leafCount = tree.nodes.iter().filter(|node| node.is_leaf).count();

    let mut leafWritten = 0usize;

    for (nodeIndex, node) in tree.nodes.iter().enumerate() {
        if !node.is_leaf {
            continue;
        }

        let polynomial = polynomials
            .iter()
            .find(|polynomial| polynomial.leaf_node_index == nodeIndex)
            .expect("Leaf polynomial not found");

        writeln!(kzgFile, "    {{").unwrap();

        writeln!(kzgFile, "      \"node_index\": {},", nodeIndex).unwrap();

        match node.parent_node {
            Some(parentIndex) => {
                writeln!(kzgFile, "      \"parent_node\": {},", parentIndex).unwrap();
            }

            None => {
                writeln!(kzgFile, "      \"parent_node\": null,").unwrap();
            }
        }

        writeln!(kzgFile, "      \"keys\": {:?},", node.keys).unwrap();

        writeln!(kzgFile, "      \"LPC\": null,").unwrap();

        writeln!(kzgFile, "      \"RPC\": null,").unwrap();

        writeln!(
            kzgFile,
            "      \"commitment\": \"{}\",",
            jsonString(format!("{:?}", polynomial.commitment))
        )
        .unwrap();

        if nodeIndex == leafIndex {
            let mut i = 0usize;

            while i < node.keys.len() && node.keys[i] != searchKey {
                i += 1;
            }

            assert!(i < node.keys.len(), "Search key not found in expected leaf");

            writeln!(
                kzgFile,
                "      \"witness\": \"{}\",",
                jsonString(format!("{:?}", witnesses[0]))
            )
            .unwrap();

            // The KZG opening point of a leaf is the key's
            // global index, not its position inside the leaf.

            writeln!(
                kzgFile,
                "      \"opening_position\": {},",
                openings[0]
                    .leaf_record
                    .as_ref()
                    .expect("Leaf record missing")
                    .global_index
            )
            .unwrap();

            writeln!(kzgFile, "      \"position_in_leaf\": {},", i).unwrap();
        } else {
            writeln!(kzgFile, "      \"witness\": null,").unwrap();

            writeln!(kzgFile, "      \"opening_position\": null,").unwrap();

            writeln!(kzgFile, "      \"position_in_leaf\": null,").unwrap();
        }

        // -------------------------------------------------
        // KZG opening of every key in this leaf
        //
        // x = global index (same as buildLeafPolynomials),
        // y = f(x), witness = computeKzgWitness(f, x).
        // -------------------------------------------------

        writeln!(kzgFile, "      \"openings\": [").unwrap();

        for position in 0..node.keys.len() {
            let evaluation = evaluations
                .iter()
                .find(|evaluation| {
                    evaluation.leaf_node_index == nodeIndex
                        && evaluation.position_in_leaf == position
                })
                .expect("Leaf evaluation not found");

            assert_eq!(
                evaluation.key, node.keys[position],
                "Leaf evaluation key mismatch at node {}",
                nodeIndex
            );

            let evaluationPoint = ark_bn254::Fr::from(evaluation.global_index as u64);

            let evaluationValue =
                leaf_kzg::evaluatePolynomial(&polynomial.coefficients, evaluationPoint);

            let leafWitness =
                leaf_kzg::computeKzgWitness(&polynomial.coefficients, evaluationPoint, srs);

            let leafOpeningVerification = verifyEval(
                &polynomial.commitment,
                evaluationPoint,
                evaluationValue,
                &leafWitness,
                srs,
            );

            assert!(
                leafOpeningVerification,
                "Leaf opening failed at node {} position {}",
                nodeIndex, position
            );

            if nodeIndex == leafIndex && node.keys[position] == searchKey {
                assert_eq!(
                    leafWitness, witnesses[0],
                    "Search-key leaf witness mismatch"
                );
            }

            writeln!(kzgFile, "        {{").unwrap();

            writeln!(kzgFile, "          \"key\": {},", node.keys[position]).unwrap();

            writeln!(kzgFile, "          \"position_in_leaf\": {},", position).unwrap();

            writeln!(
                kzgFile,
                "          \"global_index\": {},",
                evaluation.global_index
            )
            .unwrap();

            writeln!(
                kzgFile,
                "          \"evaluation_point\": \"{}\",",
                evaluationPoint
            )
            .unwrap();

            writeln!(
                kzgFile,
                "          \"evaluation_value\": \"{}\",",
                evaluationValue
            )
            .unwrap();

            writeln!(
                kzgFile,
                "          \"witness\": \"{}\",",
                jsonString(format!("{:?}", leafWitness))
            )
            .unwrap();

            writeln!(
                kzgFile,
                "          \"verification\": {}",
                leafOpeningVerification
            )
            .unwrap();

            if position + 1 < node.keys.len() {
                writeln!(kzgFile, "        }},").unwrap();
            } else {
                writeln!(kzgFile, "        }}").unwrap();
            }
        }

        writeln!(kzgFile, "      ],").unwrap();

        writeln!(kzgFile, "      \"polynomial_coefficients\": [").unwrap();

        for i in 0..polynomial.coefficients.len() {
            if i + 1 < polynomial.coefficients.len() {
                writeln!(kzgFile, "        \"{}\",", polynomial.coefficients[i]).unwrap();
            } else {
                writeln!(kzgFile, "        \"{}\"", polynomial.coefficients[i]).unwrap();
            }
        }

        writeln!(kzgFile, "      ]").unwrap();

        leafWritten += 1;

        if leafWritten < leafCount {
            writeln!(kzgFile, "    }},").unwrap();
        } else {
            writeln!(kzgFile, "    }}").unwrap();
        }
    }

    writeln!(kzgFile, "  ],").unwrap();

    // =====================================================
    // PROOF PATH
    // =====================================================

    writeln!(kzgFile, "  \"proof_path_leaf_to_root\": [").unwrap();

    let leafCommitment = nodeCommitments[leafIndex]
        .as_ref()
        .expect("Leaf commitment missing");

    writeln!(kzgFile, "    {{").unwrap();

    writeln!(kzgFile, "      \"level\": 0,").unwrap();

    writeln!(kzgFile, "      \"node_type\": \"leaf\",").unwrap();

    writeln!(kzgFile, "      \"node_index\": {},", leafIndex).unwrap();

    writeln!(
        kzgFile,
        "      \"commitment\": \"{}\",",
        jsonString(format!("{:?}", leafCommitment))
    )
    .unwrap();

    writeln!(
        kzgFile,
        "      \"witness\": \"{}\",",
        jsonString(format!("{:?}", witnesses[0]))
    )
    .unwrap();

    writeLeafRecord(
        &mut kzgFile,
        openings[0]
            .leaf_record
            .as_ref()
            .expect("Leaf record missing"),
    );

    writeVerifyEval(&mut kzgFile, &openings[0]);

    if path.is_empty() {
        writeln!(kzgFile, "    }}").unwrap();
    } else {
        writeln!(kzgFile, "    }},").unwrap();
    }

    for (level, step) in path.iter().rev().enumerate() {
        let witnessIndex = level + 1;

        let recordPosition = if step.index < step.nodeKeys.len() {
            step.index
        } else {
            step.nodeKeys.len() - 1
        };

        let mapping = if step.index < step.nodeKeys.len() {
            "LPC"
        } else {
            "RPC"
        };

        let nodeType = if step.nodeIndex == tree.root {
            "root"
        } else {
            "parent"
        };

        let commitment = nodeCommitments[step.nodeIndex]
            .as_ref()
            .expect("Path commitment missing");

        writeln!(kzgFile, "    {{").unwrap();

        writeln!(kzgFile, "      \"level\": {},", witnessIndex).unwrap();

        writeln!(kzgFile, "      \"node_type\": \"{}\",", nodeType).unwrap();

        writeln!(kzgFile, "      \"node_index\": {},", step.nodeIndex).unwrap();

        writeln!(kzgFile, "      \"record_position\": {},", recordPosition).unwrap();

        writeln!(kzgFile, "      \"mapping\": \"{}\",", mapping).unwrap();

        writeln!(
            kzgFile,
            "      \"chosen_child\": {},",
            step.chosenChildIndex
        )
        .unwrap();

        writeln!(
            kzgFile,
            "      \"commitment\": \"{}\",",
            jsonString(format!("{:?}", commitment))
        )
        .unwrap();

        writeln!(
            kzgFile,
            "      \"witness\": \"{}\",",
            jsonString(format!("{:?}", witnesses[witnessIndex]))
        )
        .unwrap();

        let opening = &openings[witnessIndex];

        writePublicPointers(
            &mut kzgFile,
            opening
                .public_pointers
                .as_ref()
                .expect("Internal-node public pointers missing"),
        );

        writeVerifyEval(&mut kzgFile, opening);

        if witnessIndex + 1 < witnesses.len() {
            writeln!(kzgFile, "    }},").unwrap();
        } else {
            writeln!(kzgFile, "    }}").unwrap();
        }
    }

    writeln!(kzgFile, "  ],").unwrap();

    // =====================================================
    // PREDECESSOR PROOF
    // =====================================================

    let predecessorOpenings = &predecessorProof.openings;

    let predecessorPath = &predecessorProof.path;

    writeln!(kzgFile, "  \"predecessor_proof\": {{").unwrap();

    writeln!(kzgFile, "    \"key\": {},", predecessorKey).unwrap();

    writeln!(
        kzgFile,
        "    \"global_index\": {},",
        predecessorProof.global_index
    )
    .unwrap();

    writeln!(
        kzgFile,
        "    \"leaf_node\": {},",
        predecessorProof.leaf_index
    )
    .unwrap();

    writeln!(
        kzgFile,
        "    \"witness_count\": {}",
        predecessorProof.witnesses.len()
    )
    .unwrap();

    writeln!(kzgFile, "  }},").unwrap();

    writeln!(kzgFile, "  \"predecessor_proof_path_leaf_to_root\": [").unwrap();

    writeln!(kzgFile, "    {{").unwrap();

    writeln!(kzgFile, "      \"level\": 0,").unwrap();

    writeln!(kzgFile, "      \"node_type\": \"leaf\",").unwrap();

    writeln!(
        kzgFile,
        "      \"node_index\": {},",
        predecessorProof.leaf_index
    )
    .unwrap();

    writeln!(
        kzgFile,
        "      \"commitment\": \"{}\",",
        jsonString(format!("{:?}", predecessorOpenings[0].commitment))
    )
    .unwrap();

    writeln!(
        kzgFile,
        "      \"witness\": \"{}\",",
        jsonString(format!("{:?}", predecessorOpenings[0].witness))
    )
    .unwrap();

    writeLeafRecord(
        &mut kzgFile,
        predecessorOpenings[0]
            .leaf_record
            .as_ref()
            .expect("Predecessor leaf record missing"),
    );

    writeVerifyEval(&mut kzgFile, &predecessorOpenings[0]);

    if predecessorPath.is_empty() {
        writeln!(kzgFile, "    }}").unwrap();
    } else {
        writeln!(kzgFile, "    }},").unwrap();
    }

    for (level, step) in predecessorPath.iter().rev().enumerate() {
        let witnessIndex = level + 1;

        let opening = &predecessorOpenings[witnessIndex];

        let recordPosition = if step.index < step.nodeKeys.len() {
            step.index
        } else {
            step.nodeKeys.len() - 1
        };

        let nodeType = if step.nodeIndex == tree.root {
            "root"
        } else {
            "parent"
        };

        writeln!(kzgFile, "    {{").unwrap();

        writeln!(kzgFile, "      \"level\": {},", witnessIndex).unwrap();

        writeln!(kzgFile, "      \"node_type\": \"{}\",", nodeType).unwrap();

        writeln!(kzgFile, "      \"node_index\": {},", step.nodeIndex).unwrap();

        writeln!(kzgFile, "      \"record_position\": {},", recordPosition).unwrap();

        writeln!(
            kzgFile,
            "      \"mapping\": \"{}\",",
            if step.index < step.nodeKeys.len() {
                "LPC"
            } else {
                "RPC"
            }
        )
        .unwrap();

        writeln!(
            kzgFile,
            "      \"chosen_child\": {},",
            step.chosenChildIndex
        )
        .unwrap();

        writeln!(
            kzgFile,
            "      \"commitment\": \"{}\",",
            jsonString(format!("{:?}", opening.commitment))
        )
        .unwrap();

        writeln!(
            kzgFile,
            "      \"witness\": \"{}\",",
            jsonString(format!("{:?}", opening.witness))
        )
        .unwrap();

        writePublicPointers(
            &mut kzgFile,
            opening
                .public_pointers
                .as_ref()
                .expect("Predecessor internal-node public pointers missing"),
        );

        writeVerifyEval(&mut kzgFile, opening);

        if witnessIndex + 1 < predecessorOpenings.len() {
            writeln!(kzgFile, "    }},").unwrap();
        } else {
            writeln!(kzgFile, "    }}").unwrap();
        }
    }

    writeln!(kzgFile, "  ],").unwrap();

    // =====================================================
    // ORDERING (keys and global indices only)
    // =====================================================

    writeln!(kzgFile, "  \"ordering\": {{").unwrap();

    writeln!(kzgFile, "    \"queried_key\": {},", searchKey).unwrap();

    writeln!(
        kzgFile,
        "    \"queried_global_index\": {},",
        searchGlobalIndex
    )
    .unwrap();

    writeln!(kzgFile, "    \"predecessor_key\": {},", predecessorKey).unwrap();

    writeln!(
        kzgFile,
        "    \"predecessor_global_index\": {}",
        predecessorProof.global_index
    )
    .unwrap();

    writeln!(kzgFile, "  }}").unwrap();

    writeln!(kzgFile, "}}").unwrap();

    println!();

    println!("JSON Files Written:");

    println!("  vplus_tree.json");

    println!("  kzg_details.json");
}
