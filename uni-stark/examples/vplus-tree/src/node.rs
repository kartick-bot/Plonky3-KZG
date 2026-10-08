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

    // These will be used later for commitments.
    pub left_pointer_commitment: Option<String>,
    pub right_pointer_commitment: Option<String>,
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
            left_pointer_commitment: None,
            right_pointer_commitment: None,
        }
    }


    // Split the children into left and right parts.
    pub fn retrieveChildren(
        &self,
        promoteIndex: usize
    ) -> (Vec<usize>, Vec<usize>) {

        // Children after the promoted key
        // start from promoteIndex + 1.
        let splitAt = promoteIndex + 1;


        // Children that stay on the left node.
        let leftChildren =
            self.children[..splitAt].to_vec();


        // Children that move to the right node.
        let rightChildren =
            self.children[splitAt..].to_vec();


        (
            leftChildren,
            rightChildren
        )
    }
}


// This controls how a node is printed.
impl fmt::Display for Node {

    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>
    ) -> fmt::Result {

        if self.is_leaf {

            write!(
                f,
                "<Leaf {:?}>",
                self.keys
            )

        } else {

            write!(
                f,
                "<Internal {:?}>",
                self.keys
            )
        }
    }
}