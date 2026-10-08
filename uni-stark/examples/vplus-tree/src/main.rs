mod node;
mod vplustree;

use vplustree::VplusTree;

// Brings shuffle() and random helpers into scope.
use rand::prelude::*;


// ---------------------------------------------------------
// Generate Random Unique Keys
// ---------------------------------------------------------

fn generateRandomKeys(
    keyCount: usize,
    universeMax: i64
) -> Vec<i64> {

    // We cannot generate more unique keys
    // than the size of the key universe.
    if keyCount > universeMax as usize {
        panic!("Key count cannot be larger than key universe.");
    }


    // Create all possible keys:
    // 1, 2, 3, ... universeMax
    let mut universe: Vec<i64> =
        (1..=universeMax).collect();


    // Create the random number generator.
    let mut rng =
        rand::rng();


    // Randomly shuffle all keys.
    universe.shuffle(&mut rng);


    // Keep only the number of keys we need.
    universe[..keyCount].to_vec()
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


    // Create the V+ Tree.
    let mut tree =
        VplusTree::new(order);


    // Insert all generated keys.
    for key in keys {

        tree.insert(
            *key,
            format!("Value{}", key)
        );
    }


    // Print the tree level by level.
    println!();
    println!("Tree Structure:");

    tree.show();


    // Print tree information.
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


    // Run basic validation.
    println!();
    println!(
        "Basic Validation: {}",
        tree.validate()
    );


    // Run full verification.
    println!();
    println!("Full Verification:");

    tree.verifyTree();
}


// ---------------------------------------------------------
// Main
// ---------------------------------------------------------

fn main() {

    // -----------------------------------------------------
    // Experiment Settings
    // -----------------------------------------------------

    // Test Order 4 and Order 5.
    let orders =
        vec![4, 5];


    // Number of keys = 5^3 = 125.
    let keyBase: usize = 5;
    let power: u32 = 3;


    // Key universe = 1 to 2^10.
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

    // Print 10 keys per line.
    // This is easier to read than one very long line.
    for chunk in keys.chunks(10) {
        println!("{:?}", chunk);
    }


    // -----------------------------------------------------
    // Build Trees
    // -----------------------------------------------------

    // Both trees use the same random keys.
    // This makes Order 4 and Order 5 comparable.
    for order in orders {

        buildTree(
            order,
            &keys
        );
    }
}