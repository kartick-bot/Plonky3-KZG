mod leaf_kzg;
mod node;
mod vplustree;

use vplustree::VplusTree;

use rand::prelude::*;
use sha2::{Digest, Sha256};
use leaf_kzg::{
    buildLeafEvaluations,
    buildLeafPolynomials,
    generateTreeSrs,
    groupEvaluationsByLeaf,
    printAndVerifyLeafPolynomials,
};


// ---------------------------------------------------------
// Generate Random Unique Keys
// ---------------------------------------------------------

fn generateRandomKeys(
    keyCount: usize,
    universeMax: i64
) -> Vec<i64> {

    if keyCount > universeMax as usize {
        panic!("Key count cannot be larger than key universe.");
    }


    let mut universe: Vec<i64> =
        (1..=universeMax).collect();


    let mut rng =
        rand::rng();


    universe.shuffle(&mut rng);


    universe[..keyCount].to_vec()
}


// ---------------------------------------------------------
// Generate SHA-256 Value For A Key
// ---------------------------------------------------------

fn generateHashValue(
    key: i64
) -> [u8; 32] {

    let input =
        format!("Value{}", key);


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
                format!("{:02x}", byte)
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
    println!("==========================================");
    println!("Search Path for Key {}", searchKey);
    println!("==========================================");


    let (leafIndex, path) =
        tree.searchPath(searchKey);


    let (_, proofPath) =
        tree.buildProofPath(searchKey);


    // -----------------------------------------------------
    // Print Search Path
    // -----------------------------------------------------

    for (level, step) in path.iter().enumerate() {

        println!();
        println!("Level {}", level);

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

            (Some(nodeIndex), Some(keys)) => {

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

            (Some(nodeIndex), Some(keys)) => {

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


        println!("          |");
        println!("          v");
    }


    // -----------------------------------------------------
    // Print Final Leaf
    // -----------------------------------------------------

    let leaf =
        &tree.nodes[leafIndex];


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
    ) in leaf.keys.iter().enumerate() {

        if *key == searchKey {

            keyPosition =
                Some(index);

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
                    &leaf.values[index]
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
    println!("==========================================");
    println!("Proof Path");
    println!("==========================================");


    for (level, step) in proofPath.iter().enumerate() {

        println!();
        println!("Level {}", level);

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

            (Some(nodeIndex), Some(keys)) => {

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

            (Some(nodeIndex), Some(keys)) => {

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
    println!("==========================================");
    println!("V+ Tree Order: {}", order);
    println!("Number of Keys: {}", keys.len());
    println!("==========================================");


    let mut tree =
        VplusTree::new(order);


    // -----------------------------------------------------
    // Insert Key-Value Pairs
    // -----------------------------------------------------

    for key in keys {

        let value =
            generateHashValue(*key);


        tree.insert(
            *key,
            value
        );
    }


    // -----------------------------------------------------
    // Print Tree
    // -----------------------------------------------------

    println!();
    println!("Tree Structure:");

    tree.show();
    let max_degree =
    order - 2;


let srs =
    generateTreeSrs(
        max_degree
    );


println!();
println!("==========================================");
println!("KZG SRS");
println!("==========================================");

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
    println!("Full Verification:");

    tree.verifyTree();
    let evaluations =
    buildLeafEvaluations(
        &tree
    );


let grouped =
    groupEvaluationsByLeaf(
        &evaluations
    );


let polynomials =
    buildLeafPolynomials(
        &grouped, 
        &srs
    );


printAndVerifyLeafPolynomials(
    &grouped,
    &polynomials
);


    // -----------------------------------------------------
    // Sample Lookups
    // -----------------------------------------------------

    println!();
    println!("Sample Key-Value Lookups:");
    println!("------------------------------------------");


    for key in keys.iter().take(5) {

        match tree.get(*key) {

            Some(value) => {

                println!(
                    "Key {} -> {}",
                    key,
                    hashToHex(&value)
                );
            }


            None => {

                println!(
                    "Key {} -> NOT FOUND",
                    key
                );
            }
        }
    }


    // -----------------------------------------------------
    // Search Path For One Key
    // -----------------------------------------------------

    let searchKey =
        keys[0];


    printSearchPath(
        &tree,
        searchKey
    );
}


// ---------------------------------------------------------
// Main
// ---------------------------------------------------------

fn main() {

    // -----------------------------------------------------
    // Experiment Settings
    // -----------------------------------------------------

    let orders =
        vec![4, 5];


    let keyBase: usize = 5;
    let power: u32 = 3;


    let universePower: u32 = 10;


    // -----------------------------------------------------
    // Calculate Values
    // -----------------------------------------------------

    let keyCount =
        keyBase.pow(power);


    let universeMax =
        2_i64.pow(universePower);


    // -----------------------------------------------------
    // Print Experiment Settings
    // -----------------------------------------------------

    println!("==========================================");
    println!("Experiment Configuration");
    println!("==========================================");

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


    println!();
    println!("Generated Random Keys:");


    for chunk in keys.chunks(10) {

        println!(
            "{:?}",
            chunk
        );
    }


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