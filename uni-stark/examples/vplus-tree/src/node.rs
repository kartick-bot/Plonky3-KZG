use ark_bn254::G1Affine;

use std::fmt;

// This represents one node in the V+ Tree.
#[derive(Debug)]
pub struct Node {
    pub order: usize,
    pub is_leaf: bool,

    // Keys stored in this node.
    pub keys: Vec<i64>,

    // Child node indexes.
    // Only internal nodes use this.
    pub children: Vec<usize>,

    // SHA-256 values stored in leaf nodes.
    // keys[i] corresponds to values[i].
    pub values: Vec<[u8; 32]>,

    // Next leaf in the linked list.
    pub next_node: Option<usize>,

    // Parent node index.
    pub parent_node: Option<usize>,

    // -----------------------------------------------------
    // Internal-node pointer commitments
    //
    // For every internal-node key:
    //
    // keys[i]
    //      <-> left_pointer_commitments[i]
    //      <-> right_pointer_commitments[i]
    //
    // So each internal key is conceptually:
    //
    // (key, LPC, RPC)
    //
    // For now the commitments are Option<G1Affine>
    // because they will be populated later.
    //
    // Leaf nodes should keep these vectors empty.
    // -----------------------------------------------------
    pub left_pointer_commitments: Vec<Option<G1Affine>>,

    pub right_pointer_commitments: Vec<Option<G1Affine>>,
}

impl Node {
    // Create a new node.
    pub fn new(order: usize, is_leaf: bool) -> Self {
        Node {
            order,
            is_leaf,

            keys: Vec::new(),

            children: Vec::new(),

            values: Vec::new(),

            next_node: None,

            parent_node: None,

            left_pointer_commitments: Vec::new(),

            right_pointer_commitments: Vec::new(),
        }
    }

    // Split the children into left and right parts.
    pub fn retrieveChildren(&self, promoteIndex: usize) -> (Vec<usize>, Vec<usize>) {
        // Children after the promoted key
        // start from promoteIndex + 1.
        let splitAt = promoteIndex + 1;

        // Children that stay on the left node.
        let leftChildren = self.children[..splitAt].to_vec();

        // Children that move to the right node.
        let rightChildren = self.children[splitAt..].to_vec();

        (leftChildren, rightChildren)
    }
}

// This controls how a node is printed.
impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_leaf {
            write!(f, "<Leaf {:?}>", self.keys)
        } else {
            write!(f, "<Internal {:?}>", self.keys)
        }
    }
}
