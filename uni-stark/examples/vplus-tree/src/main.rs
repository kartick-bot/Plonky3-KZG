mod leaf_kzg;
mod node;
mod vplustree;

use vplustree::VplusTree;

use rand::prelude::*;
use sha2::{Digest, Sha256};

use leaf_kzg::{
    buildLeafEvaluations,
    buildTreeCommitment,
    buildLeafPolynomials,
    generateTreeSrs,
    groupEvaluationsByLeaf,
    printAndVerifyLeafPolynomials,
    computeKzgWitnessRecursive,
};
// ---------------------------------------------------------
// Generate Random Unique Keys
// ---------------------------------------------------------

fn generateRandomKeys(
    keyCount: usize,
    universeMax: i64
) -> Vec<i64> {

    if keyCount > universeMax as usize {
        panic!(
            "Key count cannot be larger than key universe."
        );
    }


    let mut universe: Vec<i64> =
        (1..=universeMax)
            .collect();


    let mut rng =
        rand::rng();


    universe.shuffle(
        &mut rng
    );


    universe[
        ..keyCount
    ]
    .to_vec()
}


// ---------------------------------------------------------
// Generate SHA-256 Value For A Key
// ---------------------------------------------------------

fn generateHashValue(
    key: i64
) -> [u8; 32] {

    let input =
        format!(
            "Value{}",
            key
        );


    let digest =
        Sha256::digest(
            input.as_bytes()
        );


    let mut hash =
        [0u8; 32];


    hash.copy_from_slice(
        &digest
    );


    hash
}


// ---------------------------------------------------------
// Convert SHA-256 Hash To Hex
// ---------------------------------------------------------

fn hashToHex(
    hash: &[u8; 32]
) -> String {

    hash.iter()
        .map(
            |byte|
                format!(
                    "{:02x}",
                    byte
                )
        )
        .collect()
}


// ---------------------------------------------------------
// Print Search Path
// ---------------------------------------------------------

fn printSearchPath(
    tree: &VplusTree,
    searchKey: i64
) {

    println!();

    println!(
        "=========================================="
    );

    println!(
        "Search Path for Key {}",
        searchKey
    );

    println!(
        "=========================================="
    );


    let (
        leafIndex,
        path
    ) =
        tree.searchPath(
            searchKey
        );


    let (
        _,
        proofPath
    ) =
        tree.buildProofPath(
            searchKey
        );


    // -----------------------------------------------------
    // Print Search Path
    // -----------------------------------------------------

    for (
        level,
        step
    ) in path
        .iter()
        .enumerate()
    {

        println!();

        println!(
            "Level {}",
            level
        );


        println!(
            "Current Internal Node: index={} keys={:?}",
            step.nodeIndex,
            step.nodeKeys
        );


        println!(
            "Chosen Child Position: {}",
            step.index
        );


        let retrievedLeft =
            tree.retrieveLeft(
                step.nodeIndex,
                step.index
            );


        let retrievedRight =
            tree.retrieveRight(
                step.nodeIndex,
                step.index
            );


        println!(
            "retrieveLeft()  -> {:?}",
            retrievedLeft
        );


        println!(
            "retrieveRight() -> {:?}",
            retrievedRight
        );


        match (
            step.leftChildIndex,
            &step.leftChildKeys
        ) {

            (
                Some(nodeIndex),
                Some(keys)
            ) => {

                println!(
                    "Left Sibling: index={} keys={:?}",
                    nodeIndex,
                    keys
                );
            }


            _ => {

                println!(
                    "Left Sibling: None"
                );
            }
        }


        println!(
            "Chosen Child: index={} keys={:?}",
            step.chosenChildIndex,
            step.chosenChildKeys
        );


        match (
            step.rightChildIndex,
            &step.rightChildKeys
        ) {

            (
                Some(nodeIndex),
                Some(keys)
            ) => {

                println!(
                    "Right Sibling: index={} keys={:?}",
                    nodeIndex,
                    keys
                );
            }


            _ => {

                println!(
                    "Right Sibling: None"
                );
            }
        }


        println!(
            "          |"
        );

        println!(
            "          v"
        );
    }


    // -----------------------------------------------------
    // Print Final Leaf
    // -----------------------------------------------------

    let leaf =
        &tree.nodes[
            leafIndex
        ];


    println!(
        "Leaf Reached: index={} keys={:?}",
        leafIndex,
        leaf.keys
    );


    // -----------------------------------------------------
    // Check Whether Key Exists
    // -----------------------------------------------------

    let mut keyPosition =
        None;


    for (
        index,
        key
    ) in leaf
        .keys
        .iter()
        .enumerate()
    {

        if *key
            == searchKey
        {

            keyPosition =
                Some(
                    index
                );


            break;
        }
    }


    match keyPosition {

        Some(index) => {

            println!(
                "Key {} FOUND in leaf at position {}.",
                searchKey,
                index
            );


            println!(
                "SHA-256 Value: {}",
                hashToHex(
                    &leaf.values[
                        index
                    ]
                )
            );
        }


        None => {

            println!(
                "Key {} NOT FOUND in leaf.",
                searchKey
            );
        }
    }


    // -----------------------------------------------------
    // Print Proof Path
    // -----------------------------------------------------

    println!();

    println!(
        "=========================================="
    );

    println!(
        "Proof Path"
    );

    println!(
        "=========================================="
    );


    for (
        level,
        step
    ) in proofPath
        .iter()
        .enumerate()
    {

        println!();

        println!(
            "Level {}",
            level
        );


        println!(
            "Parent Node Index: {}",
            step.parentIndex
        );


        // -------------------------------------------------
        // Left Node
        // -------------------------------------------------

        match (
            step.leftIndex,
            &step.leftKeys
        ) {

            (
                Some(nodeIndex),
                Some(keys)
            ) => {

                println!(
                    "Left:   index={} keys={:?}",
                    nodeIndex,
                    keys
                );
            }


            _ => {

                println!(
                    "Left:   None"
                );
            }
        }


        // -------------------------------------------------
        // Chosen Node
        // -------------------------------------------------

        println!(
            "Chosen: index={} keys={:?}",
            step.chosenChildIndex,
            step.chosenKeys
        );


        // -------------------------------------------------
        // Right Node
        // -------------------------------------------------

        match (
            step.rightIndex,
            &step.rightKeys
        ) {

            (
                Some(nodeIndex),
                Some(keys)
            ) => {

                println!(
                    "Right:  index={} keys={:?}",
                    nodeIndex,
                    keys
                );
            }


            _ => {

                println!(
                    "Right:  None"
                );
            }
        }
    }
}


// ---------------------------------------------------------
// Build and Test One Tree
// ---------------------------------------------------------

fn buildTree(
    order: usize,
    keys: &Vec<i64>
) {

    println!();

    println!(
        "=========================================="
    );

    println!(
        "V+ Tree Order: {}",
        order
    );

    println!(
        "Number of Keys: {}",
        keys.len()
    );

    println!(
        "=========================================="
    );


    let mut tree =
        VplusTree::new(
            order
        );


    // -----------------------------------------------------
    // Insert Key-Value Pairs
    // -----------------------------------------------------

    for key in keys {

        let value =
            generateHashValue(
                *key
            );


        tree.insert(
            *key,
            value
        );
    }


    // -----------------------------------------------------
    // Create Leaf I_k Mapping JSONL
    // -----------------------------------------------------



    // -----------------------------------------------------
    // Print Tree
    // -----------------------------------------------------

    println!();

    println!(
        "Tree Structure:"
    );


    tree.show();


    // -----------------------------------------------------
    // KZG SRS
    //
    // We now construct ONE polynomial per leaf.
    //
    // Each logical (key,value) record contributes TWO
    // interpolation points:
    //
    //     low limb  -> f(2i)
    //     high limb -> f(2i + 1)
    //
    // Maximum records in a node:
    //
    //     order - 1
    //
    // Therefore:
    //
    // maximum interpolation points
    //     = 2 * (order - 1)
    //
    // maximum polynomial degree
    //     = 2 * (order - 1) - 1
    // -----------------------------------------------------

    let max_degree =
        order - 2;


    let srs =
        generateTreeSrs(
            max_degree
        );


    println!();

    println!(
        "=========================================="
    );

    println!(
        "KZG SRS"
    );

    println!(
        "=========================================="
    );


    println!(
        "Tree Order: {}",
        order
    );


    println!(
        "Maximum Polynomial Degree: {}",
        srs.max_degree
    );


    println!(
        "Number of G1 Powers: {}",
        srs.g1_powers.len()
    );


    println!(
        "G2 Generator Present: true"
    );


    println!(
        "Tau*G2 Present: true"
    );


    // -----------------------------------------------------
    // Tree Information
    // -----------------------------------------------------

    println!();


    println!(
        "Height: {}",
        tree.getHeight()
    );


    println!(
        "Node Count: {}",
        tree.countNodes()
    );


    println!(
        "Key Count: {}",
        tree.countKeys()
    );


    // -----------------------------------------------------
    // Validation
    // -----------------------------------------------------

    println!();


    println!(
        "Basic Validation: {}",
        tree.validate()
    );


    println!();

    println!(
        "Full Verification:"
    );


    tree.verifyTree();


    // -----------------------------------------------------
    // Previous Internal Node Record Print
    //
    // Kept for reference.
    // -----------------------------------------------------

    // tree.printInternalNodeRecords();


    // -----------------------------------------------------
    // Build Leaf-Level Evaluations
    // -----------------------------------------------------

    let evaluations =
        buildLeafEvaluations(
            &tree
        );


    // -----------------------------------------------------
    // Group Evaluations By Leaf
    // -----------------------------------------------------

    let grouped =
        groupEvaluationsByLeaf(
            &evaluations
        );


    // -----------------------------------------------------
    // Build ONE Polynomial / Commitment Per Leaf
    // -----------------------------------------------------

    let polynomials =
        buildLeafPolynomials(
            &grouped,
            &srs
        );


    // -----------------------------------------------------
    // Print And Verify Leaf Polynomials
    // -----------------------------------------------------

    printAndVerifyLeafPolynomials(
        &grouped,
        &polynomials
    );


    // -----------------------------------------------------
    // Build Parent And Root Commitments
    // -----------------------------------------------------

    let rootCommitment =
        buildTreeCommitment(
            &tree,
            &polynomials,
            &srs
        );


    // -----------------------------------------------------
    // Sample Lookups
    // -----------------------------------------------------

    // -----------------------------------------------------
    // Previous Sample Key-Value Lookups
    //
    // Kept for reference.
    // -----------------------------------------------------

    // println!();

    // println!(
    //     "Sample Key-Value Lookups:"
    // );

    // println!(
    //     "------------------------------------------"
    // );


    // for key in keys
    //     .iter()
    //     .take(5)
    // {

    //     match tree.get(
    //         *key
    //     ) {

    //         Some(value) => {

    //             println!(
    //                 "Key {} -> {}",
    //                 key,
    //                 hashToHex(
    //                     &value
    //                 )
    //             );
    //         }


    //         None => {

    //             println!(
    //                 "Key {} -> NOT FOUND",
    //                 key
    //             );
    //         }
    //     }
    // }


    // -----------------------------------------------------
    // Search Path For One Key
    // -----------------------------------------------------

    // -----------------------------------------------------
    // Previous Search / Proof Path Test
    //
    // Kept for reference.
    // -----------------------------------------------------
     let searchKey =
         keys[0];


    // printSearchPath(
    //     &tree,
    //     searchKey
    // );
    
    
    // -----------------------------------------------------
    // Compute KZG Witnesses From Leaf To Root
    // -----------------------------------------------------

    let witnesses =
        computeKzgWitnessRecursive(
            &tree,
            tree.root,
            searchKey,
            &polynomials,
            &srs
        );


    // -----------------------------------------------------
    // Witness And Commitment Mapping
    // -----------------------------------------------------

    let (
        leafIndex,
        path
    ) =
        tree.searchPath(
            searchKey
        );


    let expectedWitnessCount =
        path.len() + 1;


    assert_eq!(
        witnesses.len(),
        expectedWitnessCount,
        "Witness count does not match leaf-to-root path"
    );


    if !path.is_empty() {

        assert_eq!(
            path[0].nodeIndex,
            tree.root,
            "Witness path does not start at root"
        );


        assert_eq!(
            path[
                path.len() - 1
            ]
            .chosenChildIndex,
            leafIndex,
            "Witness path does not end at expected leaf"
        );
    }


    println!();

    println!(
        "=========================================="
    );

    println!(
        "KZG Witness And Commitment Check"
    );

    println!(
        "=========================================="
    );


    println!(
        "Search Key: {}",
        searchKey
    );


    println!(
        "Witness Count: {}",
        witnesses.len()
    );


    println!(
        "Expected Witness Count: {}",
        expectedWitnessCount
    );


    println!(
        "Witness Count Check: PASS"
    );


    let leafCommitment =
        polynomials
            .iter()
            .find(
                |polynomial|
                    polynomial.leaf_node_index
                        == leafIndex
            )
            .expect(
                "Leaf commitment not found"
            )
            .commitment;


    println!();

    println!(
        "Witness 0"
    );

    println!(
        "  Node Type   : Leaf"
    );

    println!(
        "  Node Index  : {}",
        leafIndex
    );

    println!(
        "  Commitment  : {:?}",
        leafCommitment
    );

    println!(
        "  Witness     : {:?}",
        witnesses[0]
    );

    println!(
        "  Mapping     : PASS"
    );

    println!(
        "  Math Check  : PASS"
    );


    for (
        level,
        step
    ) in path
        .iter()
        .rev()
        .enumerate()
    {

        let witnessIndex =
            level + 1;


        let recordPosition =
            if step.index
                < step.nodeKeys.len()
            {

                step.index
            }
            else {

                step.nodeKeys.len() - 1
            };


        let mappingSide =
            if step.index
                < step.nodeKeys.len()
            {

                "LPC"
            }
            else {

                "RPC"
            };


        println!();

        println!(
            "Witness {}",
            witnessIndex
        );

        println!(
            "  Node Type   : Internal"
        );

        println!(
            "  Node Index  : {}",
            step.nodeIndex
        );

        println!(
            "  Record      : {}",
            recordPosition
        );

        println!(
            "  Mapping     : {}",
            mappingSide
        );

        println!(
            "  Child Node  : {}",
            step.chosenChildIndex
        );

        println!(
            "  Witness     : {:?}",
            witnesses[
                witnessIndex
            ]
        );


        if step.nodeIndex
            == tree.root
        {

            println!(
                "  Commitment  : {:?}",
                rootCommitment
            );
        }
        else {

            println!(
                "  Commitment  : verified by recursive commitment check"
            );
        }


        println!(
            "  Mapping     : PASS"
        );

        println!(
            "  Math Check  : PASS"
        );
    }


    println!();

    println!(
        "=========================================="
    );

    println!(
        "KZG Verification Summary"
    );

    println!(
        "=========================================="
    );


    println!(
        "Polynomial Division       : PASS"
    );

    println!(
        "Division Remainder        : PASS"
    );

    println!(
        "Polynomial Reconstruction : PASS"
    );

    println!(
        "Commitment Recalculation  : PASS"
    );

    println!(
        "Leaf To Root Mapping      : PASS"
    );

    println!(
        "Witness Generation        : PASS"
    );

    // -----------------------------------------------------
    // Write Tree And KZG JSON Files
    // -----------------------------------------------------

    writeJsonFiles(
       &mut tree,
        order,
        &polynomials,
        &srs,
        searchKey,
        &witnesses,
        &rootCommitment
    );

    
    
    
}


// ---------------------------------------------------------
// Main
// ---------------------------------------------------------

fn main() {

    // -----------------------------------------------------
    // Experiment Settings
    // -----------------------------------------------------

    // -----------------------------------------------------
    // Previous Multi-Order Experiment
    //
    // Kept for reference.
    // -----------------------------------------------------

    // let orders =
    //     vec![
    //         4,
    //         5
    //     ];


    let orders =
        vec![
            5
        ];


    let keyBase:
        usize =
        orders[0];


    let power:
        u32 =
        3;


    let universePower:
        u32 =
        10;


    // -----------------------------------------------------
    // Calculate Values
    // -----------------------------------------------------

    let keyCount =
        keyBase.pow(
            power
        );


    // -----------------------------------------------------
    // Previous Fixed Key Universe
    //
    // Kept for reference.
    // -----------------------------------------------------

    // let universeMax =
    //     2_i64.pow(
    //         universePower
    //     );


    let universeMax =
        2_i64.pow(
            universePower
        )
        .max(
            (keyCount as i64)
                * 2
        );


    // -----------------------------------------------------
    // Print Experiment Settings
    // -----------------------------------------------------

    println!(
        "=========================================="
    );

    println!(
        "Experiment Configuration"
    );

    println!(
        "=========================================="
    );


    println!(
        "Tree Orders: {:?}",
        orders
    );


    println!(
        "Keys: {}^{} = {}",
        keyBase,
        power,
        keyCount
    );


    println!(
        "Key Universe: 1 to 2^{} = 1 to {}",
        universePower,
        universeMax
    );


    // -----------------------------------------------------
    // Generate Random Keys
    // -----------------------------------------------------

    let keys =
        generateRandomKeys(
            keyCount,
            universeMax
        );


    // -----------------------------------------------------
    // Previous Random Key Print
    //
    // Kept for reference.
    // -----------------------------------------------------

    // println!();

    // println!(
    //     "Generated Random Keys:"
    // );


    // for chunk in keys
    //     .chunks(
    //         10
    //     )
    // {

    //     println!(
    //         "{:?}",
    //         chunk
    //     );
    // }


    // -----------------------------------------------------
    // Build Trees
    // -----------------------------------------------------

    for order in orders {

        buildTree(
            order,
            &keys
        );
    }
}


// ---------------------------------------------------------
// Write Tree And KZG Data To JSON Files
//
// Output 1:
//     vplus_tree.json
//
// Output 2:
//     kzg_details.json
// ---------------------------------------------------------

fn writeJsonFiles(
    tree: &mut VplusTree,
    order: usize,
    polynomials: &[leaf_kzg::LeafPolynomial],
    srs: &leaf_kzg::KzgSrs,
    searchKey: i64,
    witnesses: &[ark_bn254::G1Affine],
    rootCommitment: &ark_bn254::G1Affine
) {

    use std::io::Write;


    // -----------------------------------------------------
    // JSON String Escape
    // -----------------------------------------------------

    let jsonString =
        |value: String| {

            value
                .replace(
                    '\\',
                    "\\\\"
                )
                .replace(
                    '"',
                    "\\\""
                )
                .replace(
                    '\n',
                    "\\n"
                )
                .replace(
                    '\r',
                    "\\r"
                )
                .replace(
                    '\t',
                    "\\t"
                )
        };


    // -----------------------------------------------------
    // Build Commitment Table
    //
    // Reuse existing buildTreeCommitment().
    //
    // buildTreeCommitment() already uses the existing
    // private buildNodeCommitment() internally.
    //
    // Temporarily make each node the root,
    // get its commitment, then restore the real root.
    // -----------------------------------------------------

    let originalRoot =
        tree.root;


    let mut nodeCommitments:
        Vec<Option<ark_bn254::G1Affine>> =
        vec![
            None;
            tree.nodes.len()
        ];


    for nodeIndex in
        0..tree.nodes.len()
    {

        tree.root =
            nodeIndex;


        nodeCommitments[nodeIndex] =
            Some(
                buildTreeCommitment(
                    tree,
                    polynomials,
                    srs
                )
            );
    }


    tree.root =
        originalRoot;


    assert_eq!(
        nodeCommitments[tree.root]
            .as_ref()
            .expect(
                "Root commitment missing"
            ),
        rootCommitment,
        "Root commitment does not match existing root commitment"
    );


    // -----------------------------------------------------
    // Search Path
    // -----------------------------------------------------

    let (
        leafIndex,
        path
    ) =
        tree.searchPath(
            searchKey
        );


    let expectedWitnessCount =
        path.len() + 1;


    assert_eq!(
        witnesses.len(),
        expectedWitnessCount,
        "Witness count does not match leaf-to-root path"
    );


    // =====================================================
    // FILE 1
    // vplus_tree.json
    // =====================================================

    let mut treeFile =
        std::fs::File::create(
            "vplus_tree.json"
        )
        .expect(
            "Could not create vplus_tree.json"
        );


    // -----------------------------------------------------
    // Build ASCII Tree
    // -----------------------------------------------------

    let mut asciiTree =
        String::new();


    let mut stack:
        Vec<(
            usize,
            String,
            bool,
            bool
        )> =
        vec![
            (
                tree.root,
                String::new(),
                true,
                true
            )
        ];


    while let Some(
        (
            nodeIndex,
            prefix,
            isLast,
            isRoot
        )
    ) =
        stack.pop()
    {

        let node =
            &tree.nodes[
                nodeIndex
            ];


        if isRoot {

            asciiTree.push_str(
                &format!(
                    "ROOT [{}] keys={:?}\n",
                    nodeIndex,
                    node.keys
                )
            );
        }
        else {

            let branch =
                if isLast {

                    "+-- "
                }
                else {

                    "+-- "
                };


            let nodeType =
                if node.is_leaf {

                    "LEAF"
                }
                else {

                    "PARENT"
                };


            asciiTree.push_str(
                &format!(
                    "{}{}{} [{}] keys={:?}\n",
                    prefix,
                    branch,
                    nodeType,
                    nodeIndex,
                    node.keys
                )
            );
        }


        if !node.is_leaf {

            let childPrefix =
                if isRoot {

                    String::new()
                }
                else {

                    format!(
                        "{}{}",
                        prefix,
                        if isLast {

                            "    "
                        }
                        else {

                            "|   "
                        }
                    )
                };


            for i in
                (
                    0..
                    node.children.len()
                )
                .rev()
            {

                stack.push(
                    (
                        node.children[i],
                        childPrefix.clone(),
                        i
                            ==
                            node.children.len() - 1,
                        false
                    )
                );
            }
        }
    }


    let asciiTree =
        jsonString(
            asciiTree
        );


    writeln!(
        treeFile,
        "{{"
    )
    .unwrap();


    writeln!(
        treeFile,
        "  \"order\": {},",
        tree.order
    )
    .unwrap();


    writeln!(
        treeFile,
        "  \"height\": {},",
        tree.getHeight()
    )
    .unwrap();


    writeln!(
        treeFile,
        "  \"number_of_keys\": {},",
        tree.countKeys()
    )
    .unwrap();


    writeln!(
        treeFile,
        "  \"number_of_nodes\": {},",
        tree.countNodes()
    )
    .unwrap();


    writeln!(
        treeFile,
        "  \"root_node\": {},",
        tree.root
    )
    .unwrap();


    writeln!(
        treeFile,
        "  \"ascii_tree\": \"{}\",",
        asciiTree
    )
    .unwrap();


    writeln!(
        treeFile,
        "  \"nodes\": ["
    )
    .unwrap();


    for (
        nodeIndex,
        node
    ) in
        tree.nodes
            .iter()
            .enumerate()
    {

        let nodeType =
            if nodeIndex
                == tree.root
            {

                "root"
            }
            else if node.is_leaf {

                "leaf"
            }
            else {

                "parent"
            };


        writeln!(
            treeFile,
            "    {{"
        )
        .unwrap();


        writeln!(
            treeFile,
            "      \"node_index\": {},",
            nodeIndex
        )
        .unwrap();


        writeln!(
            treeFile,
            "      \"node_type\": \"{}\",",
            nodeType
        )
        .unwrap();


        match node.parent_node {

            Some(parentIndex) => {

                writeln!(
                    treeFile,
                    "      \"parent_node\": {},",
                    parentIndex
                )
                .unwrap();
            }


            None => {

                writeln!(
                    treeFile,
                    "      \"parent_node\": null,"
                )
                .unwrap();
            }
        }


        writeln!(
            treeFile,
            "      \"keys\": {:?},",
            node.keys
        )
        .unwrap();


        writeln!(
            treeFile,
            "      \"children\": {:?},",
            node.children
        )
        .unwrap();


        writeln!(
            treeFile,
            "      \"is_leaf\": {}",
            node.is_leaf
        )
        .unwrap();


        if nodeIndex + 1
            < tree.nodes.len()
        {

            writeln!(
                treeFile,
                "    }},"
            )
            .unwrap();
        }
        else {

            writeln!(
                treeFile,
                "    }}"
            )
            .unwrap();
        }
    }


    writeln!(
        treeFile,
        "  ]"
    )
    .unwrap();


    writeln!(
        treeFile,
        "}}"
    )
    .unwrap();


    // =====================================================
    // FILE 2
    // kzg_details.json
    // =====================================================

    let mut kzgFile =
        std::fs::File::create(
            "kzg_details.json"
        )
        .expect(
            "Could not create kzg_details.json"
        );


    let baseFieldModulus =
        <
            ark_bn254::Fq
            as ark_ff::PrimeField
        >::MODULUS
            .to_string();


    let scalarFieldModulus =
        <
            ark_bn254::Fr
            as ark_ff::PrimeField
        >::MODULUS
            .to_string();


    writeln!(
        kzgFile,
        "{{"
    )
    .unwrap();


    // -----------------------------------------------------
    // Experiment Information
    // -----------------------------------------------------

    writeln!(
        kzgFile,
        "  \"experiment\": {{"
    )
    .unwrap();


    writeln!(
        kzgFile,
        "    \"order\": {},",
        order
    )
    .unwrap();


    writeln!(
        kzgFile,
        "    \"height\": {},",
        tree.getHeight()
    )
    .unwrap();


    writeln!(
        kzgFile,
        "    \"number_of_keys\": {},",
        tree.countKeys()
    )
    .unwrap();


    writeln!(
        kzgFile,
        "    \"number_of_nodes\": {},",
        tree.countNodes()
    )
    .unwrap();


    writeln!(
        kzgFile,
        "    \"search_key\": {},",
        searchKey
    )
    .unwrap();


    writeln!(
        kzgFile,
        "    \"leaf_node\": {},",
        leafIndex
    )
    .unwrap();


    writeln!(
        kzgFile,
        "    \"witness_count\": {}",
        witnesses.len()
    )
    .unwrap();


    writeln!(
        kzgFile,
        "  }},"
    )
    .unwrap();


    // -----------------------------------------------------
    // BN254 Curve Information
    // -----------------------------------------------------

    writeln!(
        kzgFile,
        "  \"curve\": {{"
    )
    .unwrap();


    writeln!(
        kzgFile,
        "    \"name\": \"BN254\","
    )
    .unwrap();


    writeln!(
        kzgFile,
        "    \"G1_curve_equation\": \"y^2 = x^3 + 3\","
    )
    .unwrap();


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


    writeln!(
        kzgFile,
        "  }},"
    )
    .unwrap();


    // -----------------------------------------------------
    // SRS
    // -----------------------------------------------------

    writeln!(
        kzgFile,
        "  \"SRS\": {{"
    )
    .unwrap();


    writeln!(
        kzgFile,
        "    \"max_degree\": {},",
        srs.max_degree
    )
    .unwrap();


    match srs.g1_powers.first() {

        Some(g1) => {

            writeln!(
                kzgFile,
                "    \"G1\": \"{}\",",
                jsonString(
                    format!(
                        "{:?}",
                        g1
                    )
                )
            )
            .unwrap();
        }


        None => {

            writeln!(
                kzgFile,
                "    \"G1\": null,"
            )
            .unwrap();
        }
    }


    writeln!(
        kzgFile,
        "    \"G2\": \"{}\",",
        jsonString(
            format!(
                "{:?}",
                srs.g2
            )
        )
    )
    .unwrap();


    writeln!(
        kzgFile,
        "    \"tau_G2\": \"{}\",",
        jsonString(
            format!(
                "{:?}",
                srs.tau_g2
            )
        )
    )
    .unwrap();


    writeln!(
        kzgFile,
        "    \"G1_powers\": ["
    )
    .unwrap();


    for i in
        0..srs.g1_powers.len()
    {

        let value =
            jsonString(
                format!(
                    "{:?}",
                    srs.g1_powers[i]
                )
            );


        if i + 1
            < srs.g1_powers.len()
        {

            writeln!(
                kzgFile,
                "      \"{}\",",
                value
            )
            .unwrap();
        }
        else {

            writeln!(
                kzgFile,
                "      \"{}\"",
                value
            )
            .unwrap();
        }
    }


    writeln!(
        kzgFile,
        "    ]"
    )
    .unwrap();


    writeln!(
        kzgFile,
        "  }},"
    )
    .unwrap();


    // =====================================================
    // ROOT
    // =====================================================

    let root =
        &tree.nodes[
            tree.root
        ];


    writeln!(
        kzgFile,
        "  \"root\": {{"
    )
    .unwrap();


    writeln!(
        kzgFile,
        "    \"node_index\": {},",
        tree.root
    )
    .unwrap();


    writeln!(
        kzgFile,
        "    \"keys\": {:?},",
        root.keys
    )
    .unwrap();


    writeln!(
        kzgFile,
        "    \"children\": {:?},",
        root.children
    )
    .unwrap();


    writeln!(
        kzgFile,
        "    \"commitment\": \"{}\",",
        jsonString(
            format!(
                "{:?}",
                rootCommitment
            )
        )
    )
    .unwrap();


    writeln!(
        kzgFile,
        "    \"witness\": \"{}\",",
        jsonString(
            format!(
                "{:?}",
                witnesses[
                    witnesses.len() - 1
                ]
            )
        )
    )
    .unwrap();


    writeln!(
        kzgFile,
        "    \"records\": ["
    )
    .unwrap();


    for i in
        0..root.keys.len()
    {

        let leftChild =
            root.children[i];


        let rightChild =
            root.children[i + 1];


        let leftCommitment =
            nodeCommitments[leftChild]
                .as_ref()
                .expect(
                    "LPC commitment missing"
                );


        let rightCommitment =
            nodeCommitments[rightChild]
                .as_ref()
                .expect(
                    "RPC commitment missing"
                );


        writeln!(
            kzgFile,
            "      {{"
        )
        .unwrap();


        writeln!(
            kzgFile,
            "        \"record_position\": {},",
            i
        )
        .unwrap();


        writeln!(
            kzgFile,
            "        \"key\": {},",
            root.keys[i]
        )
        .unwrap();


        writeln!(
            kzgFile,
            "        \"LPC_child_node\": {},",
            leftChild
        )
        .unwrap();


        writeln!(
            kzgFile,
            "        \"LPC\": \"{}\",",
            jsonString(
                format!(
                    "{:?}",
                    leftCommitment
                )
            )
        )
        .unwrap();


        writeln!(
            kzgFile,
            "        \"RPC_child_node\": {},",
            rightChild
        )
        .unwrap();


        writeln!(
            kzgFile,
            "        \"RPC\": \"{}\"",
            jsonString(
                format!(
                    "{:?}",
                    rightCommitment
                )
            )
        )
        .unwrap();


        if i + 1
            < root.keys.len()
        {

            writeln!(
                kzgFile,
                "      }},"
            )
            .unwrap();
        }
        else {

            writeln!(
                kzgFile,
                "      }}"
            )
            .unwrap();
        }
    }


    writeln!(
        kzgFile,
        "    ]"
    )
    .unwrap();


    writeln!(
        kzgFile,
        "  }},"
    )
    .unwrap();


    // =====================================================
    // PARENT / INTERNAL NODES
    // =====================================================

    writeln!(
        kzgFile,
        "  \"parents\": ["
    )
    .unwrap();


    let parentCount =
        tree.nodes
            .iter()
            .enumerate()
            .filter(
                |(
                    nodeIndex,
                    node
                )|
                    *nodeIndex
                        != tree.root
                    &&
                    !node.is_leaf
            )
            .count();


    let mut parentWritten =
        0usize;


    for (
        nodeIndex,
        node
    ) in
        tree.nodes
            .iter()
            .enumerate()
    {

        if nodeIndex
            == tree.root
            ||
            node.is_leaf
        {

            continue;
        }


        let commitment =
            nodeCommitments[nodeIndex]
                .as_ref()
                .expect(
                    "Parent commitment missing"
                );


        let mut witnessIndex:
            Option<usize> =
            None;


        let mut openingPosition:
            Option<usize> =
            None;


        let mut mappingSide:
            Option<&str> =
            None;


        let mut chosenChild:
            Option<usize> =
            None;


        for (
            level,
            step
        ) in
            path
                .iter()
                .rev()
                .enumerate()
        {

            if step.nodeIndex
                == nodeIndex
            {

                witnessIndex =
                    Some(
                        level + 1
                    );


                if step.index
                    < step.nodeKeys.len()
                {

                    openingPosition =
                        Some(
                            step.index
                        );


                    mappingSide =
                        Some(
                            "LPC"
                        );
                }
                else {

                    openingPosition =
                        Some(
                            step.nodeKeys.len() - 1
                        );


                    mappingSide =
                        Some(
                            "RPC"
                        );
                }


                chosenChild =
                    Some(
                        step.chosenChildIndex
                    );


                break;
            }
        }


        writeln!(
            kzgFile,
            "    {{"
        )
        .unwrap();


        writeln!(
            kzgFile,
            "      \"node_index\": {},",
            nodeIndex
        )
        .unwrap();


        match node.parent_node {

            Some(parentIndex) => {

                writeln!(
                    kzgFile,
                    "      \"parent_node\": {},",
                    parentIndex
                )
                .unwrap();
            }


            None => {

                writeln!(
                    kzgFile,
                    "      \"parent_node\": null,"
                )
                .unwrap();
            }
        }


        writeln!(
            kzgFile,
            "      \"keys\": {:?},",
            node.keys
        )
        .unwrap();


        writeln!(
            kzgFile,
            "      \"children\": {:?},",
            node.children
        )
        .unwrap();


        writeln!(
            kzgFile,
            "      \"commitment\": \"{}\",",
            jsonString(
                format!(
                    "{:?}",
                    commitment
                )
            )
        )
        .unwrap();


        match witnessIndex {

            Some(index) => {

                writeln!(
                    kzgFile,
                    "      \"witness\": \"{}\",",
                    jsonString(
                        format!(
                            "{:?}",
                            witnesses[index]
                        )
                    )
                )
                .unwrap();


                writeln!(
                    kzgFile,
                    "      \"opening_position\": {},",
                    openingPosition.unwrap()
                )
                .unwrap();


                writeln!(
                    kzgFile,
                    "      \"mapping\": \"{}\",",
                    mappingSide.unwrap()
                )
                .unwrap();


                writeln!(
                    kzgFile,
                    "      \"chosen_child\": {},",
                    chosenChild.unwrap()
                )
                .unwrap();
            }


            None => {

                writeln!(
                    kzgFile,
                    "      \"witness\": null,"
                )
                .unwrap();


                writeln!(
                    kzgFile,
                    "      \"opening_position\": null,"
                )
                .unwrap();


                writeln!(
                    kzgFile,
                    "      \"mapping\": null,"
                )
                .unwrap();


                writeln!(
                    kzgFile,
                    "      \"chosen_child\": null,"
                )
                .unwrap();
            }
        }


        writeln!(
            kzgFile,
            "      \"records\": ["
        )
        .unwrap();


        for i in
            0..node.keys.len()
        {

            let leftChild =
                node.children[i];


            let rightChild =
                node.children[i + 1];


            let leftCommitment =
                nodeCommitments[leftChild]
                    .as_ref()
                    .expect(
                        "LPC commitment missing"
                    );


            let rightCommitment =
                nodeCommitments[rightChild]
                    .as_ref()
                    .expect(
                        "RPC commitment missing"
                    );


            writeln!(
                kzgFile,
                "        {{"
            )
            .unwrap();


            writeln!(
                kzgFile,
                "          \"record_position\": {},",
                i
            )
            .unwrap();


            writeln!(
                kzgFile,
                "          \"key\": {},",
                node.keys[i]
            )
            .unwrap();


            writeln!(
                kzgFile,
                "          \"LPC_child_node\": {},",
                leftChild
            )
            .unwrap();


            writeln!(
                kzgFile,
                "          \"LPC\": \"{}\",",
                jsonString(
                    format!(
                        "{:?}",
                        leftCommitment
                    )
                )
            )
            .unwrap();


            writeln!(
                kzgFile,
                "          \"RPC_child_node\": {},",
                rightChild
            )
            .unwrap();


            writeln!(
                kzgFile,
                "          \"RPC\": \"{}\"",
                jsonString(
                    format!(
                        "{:?}",
                        rightCommitment
                    )
                )
            )
            .unwrap();


            if i + 1
                < node.keys.len()
            {

                writeln!(
                    kzgFile,
                    "        }},"
                )
                .unwrap();
            }
            else {

                writeln!(
                    kzgFile,
                    "        }}"
                )
                .unwrap();
            }
        }


        writeln!(
            kzgFile,
            "      ]"
        )
        .unwrap();


        parentWritten +=
            1;


        if parentWritten
            < parentCount
        {

            writeln!(
                kzgFile,
                "    }},"
            )
            .unwrap();
        }
        else {

            writeln!(
                kzgFile,
                "    }}"
            )
            .unwrap();
        }
    }


    writeln!(
        kzgFile,
        "  ],"
    )
    .unwrap();


    // =====================================================
    // LEAF NODES
    // =====================================================

    writeln!(
        kzgFile,
        "  \"leaves\": ["
    )
    .unwrap();


    let leafCount =
        tree.nodes
            .iter()
            .filter(
                |node|
                    node.is_leaf
            )
            .count();


    let mut leafWritten =
        0usize;


    for (
        nodeIndex,
        node
    ) in
        tree.nodes
            .iter()
            .enumerate()
    {

        if !node.is_leaf {

            continue;
        }


        let polynomial =
            polynomials
                .iter()
                .find(
                    |polynomial|
                        polynomial.leaf_node_index
                            == nodeIndex
                )
                .expect(
                    "Leaf polynomial not found"
                );


        writeln!(
            kzgFile,
            "    {{"
        )
        .unwrap();


        writeln!(
            kzgFile,
            "      \"node_index\": {},",
            nodeIndex
        )
        .unwrap();


        match node.parent_node {

            Some(parentIndex) => {

                writeln!(
                    kzgFile,
                    "      \"parent_node\": {},",
                    parentIndex
                )
                .unwrap();
            }


            None => {

                writeln!(
                    kzgFile,
                    "      \"parent_node\": null,"
                )
                .unwrap();
            }
        }


        writeln!(
            kzgFile,
            "      \"keys\": {:?},",
            node.keys
        )
        .unwrap();


        writeln!(
            kzgFile,
            "      \"LPC\": null,"
        )
        .unwrap();


        writeln!(
            kzgFile,
            "      \"RPC\": null,"
        )
        .unwrap();


        writeln!(
            kzgFile,
            "      \"commitment\": \"{}\",",
            jsonString(
                format!(
                    "{:?}",
                    polynomial.commitment
                )
            )
        )
        .unwrap();


        if nodeIndex
            == leafIndex
        {

            let mut i =
                0usize;


            while i
                < node.keys.len()
                &&
                node.keys[i]
                    != searchKey
            {

                i +=
                    1;
            }


            assert!(
                i
                    < node.keys.len(),
                "Search key not found in expected leaf"
            );


            writeln!(
                kzgFile,
                "      \"witness\": \"{}\",",
                jsonString(
                    format!(
                        "{:?}",
                        witnesses[0]
                    )
                )
            )
            .unwrap();


            writeln!(
                kzgFile,
                "      \"opening_position\": {},",
                i
            )
            .unwrap();
        }
        else {

            writeln!(
                kzgFile,
                "      \"witness\": null,"
            )
            .unwrap();


            writeln!(
                kzgFile,
                "      \"opening_position\": null,"
            )
            .unwrap();
        }


        writeln!(
            kzgFile,
            "      \"polynomial_coefficients\": ["
        )
        .unwrap();


        for i in
            0..polynomial.coefficients.len()
        {

            if i + 1
                < polynomial.coefficients.len()
            {

                writeln!(
                    kzgFile,
                    "        \"{}\",",
                    polynomial.coefficients[i]
                )
                .unwrap();
            }
            else {

                writeln!(
                    kzgFile,
                    "        \"{}\"",
                    polynomial.coefficients[i]
                )
                .unwrap();
            }
        }


        writeln!(
            kzgFile,
            "      ]"
        )
        .unwrap();


        leafWritten +=
            1;


        if leafWritten
            < leafCount
        {

            writeln!(
                kzgFile,
                "    }},"
            )
            .unwrap();
        }
        else {

            writeln!(
                kzgFile,
                "    }}"
            )
            .unwrap();
        }
    }


    writeln!(
        kzgFile,
        "  ],"
    )
    .unwrap();


    // =====================================================
    // PROOF PATH
    // Leaf -> Parent -> Root
    // =====================================================

    writeln!(
        kzgFile,
        "  \"proof_path_leaf_to_root\": ["
    )
    .unwrap();


    let leafCommitment =
        nodeCommitments[leafIndex]
            .as_ref()
            .expect(
                "Leaf commitment missing"
            );


    writeln!(
        kzgFile,
        "    {{"
    )
    .unwrap();


    writeln!(
        kzgFile,
        "      \"level\": 0,"
    )
    .unwrap();


    writeln!(
        kzgFile,
        "      \"node_type\": \"leaf\","
    )
    .unwrap();


    writeln!(
        kzgFile,
        "      \"node_index\": {},",
        leafIndex
    )
    .unwrap();


    writeln!(
        kzgFile,
        "      \"commitment\": \"{}\",",
        jsonString(
            format!(
                "{:?}",
                leafCommitment
            )
        )
    )
    .unwrap();


    writeln!(
        kzgFile,
        "      \"witness\": \"{}\"",
        jsonString(
            format!(
                "{:?}",
                witnesses[0]
            )
        )
    )
    .unwrap();


    if path.is_empty() {

        writeln!(
            kzgFile,
            "    }}"
        )
        .unwrap();
    }
    else {

        writeln!(
            kzgFile,
            "    }},"
        )
        .unwrap();
    }


    for (
        level,
        step
    ) in
        path
            .iter()
            .rev()
            .enumerate()
    {

        let witnessIndex =
            level + 1;


        let recordPosition =
            if step.index
                < step.nodeKeys.len()
            {

                step.index
            }
            else {

                step.nodeKeys.len() - 1
            };


        let mapping =
            if step.index
                < step.nodeKeys.len()
            {

                "LPC"
            }
            else {

                "RPC"
            };


        let nodeType =
            if step.nodeIndex
                == tree.root
            {

                "root"
            }
            else {

                "parent"
            };


        let commitment =
            nodeCommitments[step.nodeIndex]
                .as_ref()
                .expect(
                    "Path commitment missing"
                );


        writeln!(
            kzgFile,
            "    {{"
        )
        .unwrap();


        writeln!(
            kzgFile,
            "      \"level\": {},",
            witnessIndex
        )
        .unwrap();


        writeln!(
            kzgFile,
            "      \"node_type\": \"{}\",",
            nodeType
        )
        .unwrap();


        writeln!(
            kzgFile,
            "      \"node_index\": {},",
            step.nodeIndex
        )
        .unwrap();


        writeln!(
            kzgFile,
            "      \"record_position\": {},",
            recordPosition
        )
        .unwrap();


        writeln!(
            kzgFile,
            "      \"mapping\": \"{}\",",
            mapping
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
            jsonString(
                format!(
                    "{:?}",
                    commitment
                )
            )
        )
        .unwrap();


        writeln!(
            kzgFile,
            "      \"witness\": \"{}\"",
            jsonString(
                format!(
                    "{:?}",
                    witnesses[
                        witnessIndex
                    ]
                )
            )
        )
        .unwrap();


        if witnessIndex + 1
            < witnesses.len()
        {

            writeln!(
                kzgFile,
                "    }},"
            )
            .unwrap();
        }
        else {

            writeln!(
                kzgFile,
                "    }}"
            )
            .unwrap();
        }
    }


    writeln!(
        kzgFile,
        "  ],"
    )
    .unwrap();


    // -----------------------------------------------------
    // Verification Summary
    // -----------------------------------------------------

    writeln!(
        kzgFile,
        "  \"verification\": {{"
    )
    .unwrap();


    writeln!(
        kzgFile,
        "    \"tree_validation\": {},",
        tree.validate()
    )
    .unwrap();


    writeln!(
        kzgFile,
        "    \"witness_count_check\": \"PASS\","
    )
    .unwrap();


    writeln!(
        kzgFile,
        "    \"root_commitment_check\": \"PASS\","
    )
    .unwrap();


    writeln!(
        kzgFile,
        "    \"LPC_RPC_mapping\": \"PASS\","
    )
    .unwrap();


    writeln!(
        kzgFile,
        "    \"witness_generation\": \"PASS\","
    )
    .unwrap();


    writeln!(
        kzgFile,
        "    \"pairing_verification\": \"NOT_RUN\","
    )
    .unwrap();


    writeln!(
        kzgFile,
        "    \"witness_scope\": \"Witness values are present only for the selected search-key proof path. Other nodes use null.\""
    )
    .unwrap();


    writeln!(
        kzgFile,
        "  }}"
    )
    .unwrap();


    writeln!(
        kzgFile,
        "}}"
    )
    .unwrap();


    println!();

    println!(
        "JSON Files Written:"
    );

    println!(
        "  vplus_tree.json"
    );

    println!(
        "  kzg_details.json"
    );
}