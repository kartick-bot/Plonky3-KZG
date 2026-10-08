use crate::node::Node;


// ---------------------------------------------------------
// Search Path Structure
// ---------------------------------------------------------

#[derive(Debug)]
pub struct SearchPathStep {
    pub nodeKeys: Vec<i64>,
    pub index: usize,
    pub leftChildKeys: Option<Vec<i64>>,
    pub chosenChildKeys: Vec<i64>,
    pub rightChildKeys: Option<Vec<i64>>,
}


// ---------------------------------------------------------
// V+ Tree Structure
// ---------------------------------------------------------

pub struct VplusTree {
    pub order: usize,

    // Index of root node inside nodes
    pub root: usize,

    // All nodes are stored here
    pub nodes: Vec<Node>,
}


// ---------------------------------------------------------
// V+ Tree Implementation
// ---------------------------------------------------------

impl VplusTree {

    // ---------------------------------------------------------
    // Constructor
    // ---------------------------------------------------------

    pub fn new(order: usize) -> Self {

        if order < 3 {
            panic!("Order must be at least 3");
        }

        let rootNode = Node::new(order, true);

        VplusTree {
            order,
            root: 0,
            nodes: vec![rootNode],
        }
    }


    // ---------------------------------------------------------
    // Maximum Keys
    // ---------------------------------------------------------

    pub fn maxKeys(&self) -> usize {
        self.order - 1
    }


    // ---------------------------------------------------------
    // Minimum Internal Keys
    // ---------------------------------------------------------

    pub fn minInternalKeys(&self) -> usize {
        (self.order - 1) / 2
    }


    // ---------------------------------------------------------
    // Minimum Leaf Keys
    // ---------------------------------------------------------

    pub fn minLeafKeys(&self) -> usize {
        self.order / 2
    }


    // ---------------------------------------------------------
    // Underflow Check
    // ---------------------------------------------------------

    pub fn isUnderflow(&self, nodeIndex: usize) -> bool {

        let node = &self.nodes[nodeIndex];

        if nodeIndex == self.root {
            node.keys.len() < 1
        }
        else if node.is_leaf {
            node.keys.len() < self.minLeafKeys()
        }
        else {
            node.keys.len() < self.minInternalKeys()
        }
    }


    // ---------------------------------------------------------
    // Unified Split Helper
    // ---------------------------------------------------------

    pub fn splitNodeHelper(node: &mut Node) -> (Node, i64) {

        let order = node.order;


        // -----------------------------------------------------
        // Leaf Split
        // -----------------------------------------------------

        if node.is_leaf {

            let midIndex = order / 2;

            let mut rightNode = Node::new(order, true);


            rightNode.keys =
                node.keys[midIndex..].to_vec();

            rightNode.values =
                node.values[midIndex..].to_vec();


            node.keys =
                node.keys[..midIndex].to_vec();

            node.values =
                node.values[..midIndex].to_vec();


            rightNode.next_node =
                node.next_node;

            rightNode.parent_node =
                node.parent_node;


            let promoteKey =
                rightNode.keys[0];


            return (rightNode, promoteKey);
        }


        // -----------------------------------------------------
        // Internal Node Split
        // -----------------------------------------------------

        let midIndex =
            (order - 1) / 2;


        let promoteKey =
            node.keys[midIndex];


        let (leftChildren, rightChildren) =
            node.retrieveChildren(midIndex);


        let mut rightNode =
            Node::new(order, false);


        rightNode.keys =
            node.keys[midIndex + 1..].to_vec();


        rightNode.children =
            rightChildren;


        node.keys =
            node.keys[..midIndex].to_vec();


        node.children =
            leftChildren;


        rightNode.parent_node =
            node.parent_node;


        (rightNode, promoteKey)
    }


    // ---------------------------------------------------------
    // Search
    // ---------------------------------------------------------

    pub fn search(&self, key: i64) -> usize {

        let mut currentNode =
            self.root;


        while !self.nodes[currentNode].is_leaf {

            let mut index = 0;


            // Same behavior as Python bisect_right
            while index < self.nodes[currentNode].keys.len()
                && key >= self.nodes[currentNode].keys[index]
            {
                index += 1;
            }


            currentNode =
                self.nodes[currentNode].children[index];
        }


        currentNode
    }


    // ---------------------------------------------------------
    // Search Path
    // ---------------------------------------------------------

    pub fn searchPath(
        &self,
        key: i64
    ) -> (usize, Vec<SearchPathStep>) {

        let mut path =
            Vec::new();


        let mut currentNode =
            self.root;


        while !self.nodes[currentNode].is_leaf {

            let mut index = 0;


            while index < self.nodes[currentNode].keys.len()
                && key >= self.nodes[currentNode].keys[index]
            {
                index += 1;
            }


            let chosenChild =
                self.nodes[currentNode].children[index];


            let leftChildKeys =
                if index > 0 {

                    let leftChild =
                        self.nodes[currentNode].children[index - 1];

                    Some(
                        self.nodes[leftChild]
                            .keys
                            .clone()
                    )
                }
                else {
                    None
                };


            let rightChildKeys =
                if index + 1
                    < self.nodes[currentNode].children.len()
                {

                    let rightChild =
                        self.nodes[currentNode].children[index + 1];

                    Some(
                        self.nodes[rightChild]
                            .keys
                            .clone()
                    )
                }
                else {
                    None
                };


            path.push(
                SearchPathStep {

                    nodeKeys:
                        self.nodes[currentNode]
                            .keys
                            .clone(),

                    index,

                    leftChildKeys,

                    chosenChildKeys:
                        self.nodes[chosenChild]
                            .keys
                            .clone(),

                    rightChildKeys,
                }
            );


            currentNode =
                chosenChild;
        }


        (currentNode, path)
    }


    // ---------------------------------------------------------
    // Get
    // ---------------------------------------------------------

    pub fn get(&self, key: i64) -> Option<String> {

        let leafIndex =
            self.search(key);


        let leaf =
            &self.nodes[leafIndex];


        let mut index = 0;


        // Same behavior as Python bisect_left
        while index < leaf.keys.len()
            && leaf.keys[index] < key
        {
            index += 1;
        }


        if index < leaf.keys.len()
            && leaf.keys[index] == key
        {
            return Some(
                leaf.values[index].clone()
            );
        }


        None
    }


    // ---------------------------------------------------------
    // Update
    // ---------------------------------------------------------

    pub fn update(
        &mut self,
        key: i64,
        value: String
    ) -> bool {

        let leafIndex =
            self.search(key);


        let mut index = 0;


        while index < self.nodes[leafIndex].keys.len()
            && self.nodes[leafIndex].keys[index] < key
        {
            index += 1;
        }


        if index < self.nodes[leafIndex].keys.len()
            && self.nodes[leafIndex].keys[index] == key
        {

            self.nodes[leafIndex]
                .values[index] = value;

            return true;
        }


        false
    }


    // ---------------------------------------------------------
    // Insert
    // ---------------------------------------------------------

    pub fn insert(
        &mut self,
        key: i64,
        value: String
    ) {

        let leafIndex =
            self.search(key);


        let mut index = 0;


        // Same behavior as Python bisect_left
        while index < self.nodes[leafIndex].keys.len()
            && self.nodes[leafIndex].keys[index] < key
        {
            index += 1;
        }


        // Key already exists
        if index < self.nodes[leafIndex].keys.len()
            && self.nodes[leafIndex].keys[index] == key
        {

            self.nodes[leafIndex]
                .values[index] = value;

            return;
        }


        // Insert key
        self.nodes[leafIndex]
            .keys
            .insert(index, key);


        // Insert value
        self.nodes[leafIndex]
            .values
            .insert(index, value);


        // Split if leaf overflows
        if self.nodes[leafIndex].keys.len()
            > self.maxKeys()
        {
            self.splitLeaf(leafIndex);
        }
    }


    // ---------------------------------------------------------
    // Leaf Split
    // ---------------------------------------------------------

    pub fn splitLeaf(
        &mut self,
        leafIndex: usize
    ) {

        let parentIndex =
            self.nodes[leafIndex].parent_node;


        let (mut rightNode, promoteKey) = {

            let leaf =
                &mut self.nodes[leafIndex];

            VplusTree::splitNodeHelper(leaf)
        };


        let rightIndex =
            self.nodes.len();


        rightNode.parent_node =
            parentIndex;


        self.nodes.push(rightNode);


        // Connect the leaf linked list
        self.nodes[leafIndex].next_node =
            Some(rightIndex);


        // Root leaf
        if parentIndex.is_none() {

            self.createNewRoot(
                leafIndex,
                rightIndex,
                promoteKey
            );
        }
        else {

            self.insertInternal(
                parentIndex.unwrap(),
                promoteKey,
                rightIndex
            );
        }
    }


    // ---------------------------------------------------------
    // Insert Internal
    // ---------------------------------------------------------

    pub fn insertInternal(
        &mut self,
        parentIndex: usize,
        key: i64,
        childIndex: usize
    ) {

        let mut index = 0;


        // Same behavior as Python bisect_right
        while index < self.nodes[parentIndex].keys.len()
            && key >= self.nodes[parentIndex].keys[index]
        {
            index += 1;
        }


        self.nodes[parentIndex]
            .keys
            .insert(index, key);


        self.nodes[parentIndex]
            .children
            .insert(index + 1, childIndex);


        self.nodes[childIndex].parent_node =
            Some(parentIndex);


        // Split parent if it overflows
        if self.nodes[parentIndex].keys.len()
            > self.maxKeys()
        {
            self.splitInternal(parentIndex);
        }
    }


    // ---------------------------------------------------------
    // Internal Split
    // ---------------------------------------------------------

    pub fn splitInternal(
        &mut self,
        nodeIndex: usize
    ) {

        let parentIndex =
            self.nodes[nodeIndex].parent_node;


        let (mut rightNode, promoteKey) = {

            let node =
                &mut self.nodes[nodeIndex];

            VplusTree::splitNodeHelper(node)
        };


        let rightIndex =
            self.nodes.len();


        rightNode.parent_node =
            parentIndex;


        let rightChildren =
            rightNode.children.clone();


        self.nodes.push(rightNode);


        // Children moved to the right node
        // must point to their new parent
        for childIndex in rightChildren {

            self.nodes[childIndex].parent_node =
                Some(rightIndex);
        }


        // Node was the root
        if parentIndex.is_none() {

            self.createNewRoot(
                nodeIndex,
                rightIndex,
                promoteKey
            );
        }
        else {

            self.insertInternal(
                parentIndex.unwrap(),
                promoteKey,
                rightIndex
            );
        }
    }


    // ---------------------------------------------------------
    // Create New Root
    // ---------------------------------------------------------

    pub fn createNewRoot(
        &mut self,
        leftChild: usize,
        rightChild: usize,
        key: i64
    ) {

        let mut newRoot =
            Node::new(self.order, false);


        newRoot.keys.push(key);


        newRoot.children.push(leftChild);
        newRoot.children.push(rightChild);


        let newRootIndex =
            self.nodes.len();


        self.nodes.push(newRoot);


        self.nodes[leftChild].parent_node =
            Some(newRootIndex);


        self.nodes[rightChild].parent_node =
            Some(newRootIndex);


        self.root =
            newRootIndex;
    }


    // ---------------------------------------------------------
    // Show Tree
    // ---------------------------------------------------------

    pub fn show(&self) {

        let mut currentLevel =
            vec![self.root];

        let mut depth = 0;


        while !currentLevel.is_empty() {

            let mut nextLevel =
                Vec::new();


            print!("Level {}: ", depth);


            for nodeIndex in &currentLevel {

                let node =
                    &self.nodes[*nodeIndex];


                if node.is_leaf {

                    print!(
                        "<Leaf {:?}> ",
                        node.keys
                    );
                }
                else {

                    print!(
                        "<Internal {:?}> ",
                        node.keys
                    );


                    for childIndex in &node.children {

                        nextLevel.push(
                            *childIndex
                        );
                    }
                }
            }


            println!();


            currentLevel =
                nextLevel;

            depth += 1;
        }
    }


    // ---------------------------------------------------------
    // Height
    // ---------------------------------------------------------

    pub fn getHeight(&self) -> usize {

        let mut height = 0;

        let mut currentNode =
            self.root;


        while !self.nodes[currentNode].is_leaf {

            height += 1;

            currentNode =
                self.nodes[currentNode].children[0];
        }


        height
    }


    // ---------------------------------------------------------
    // Depth
    // ---------------------------------------------------------

    pub fn getDepth(
        &self,
        nodeIndex: usize
    ) -> usize {

        let mut depth = 0;

        let mut currentNode =
            nodeIndex;


        while let Some(parentIndex) =
            self.nodes[currentNode].parent_node
        {

            depth += 1;

            currentNode =
                parentIndex;
        }


        depth
    }


    // ---------------------------------------------------------
    // Count Nodes
    // ---------------------------------------------------------

    pub fn countNodes(&self) -> usize {

        let mut count = 0;

        let mut queue =
            vec![self.root];


        while !queue.is_empty() {

            let nodeIndex =
                queue.remove(0);


            count += 1;


            if !self.nodes[nodeIndex].is_leaf {

                for childIndex in
                    &self.nodes[nodeIndex].children
                {

                    queue.push(
                        *childIndex
                    );
                }
            }
        }


        count
    }


    // ---------------------------------------------------------
    // Get All Leaves
    // ---------------------------------------------------------

    pub fn getAllLeaves(&self) -> Vec<usize> {

        let mut leaves =
            Vec::new();


        let mut currentNode =
            self.root;


        // Find left-most leaf
        while !self.nodes[currentNode].is_leaf {

            currentNode =
                self.nodes[currentNode].children[0];
        }


        // Follow leaf pointers
        loop {

            leaves.push(
                currentNode
            );


            match self.nodes[currentNode].next_node {

                Some(nextNode) => {

                    currentNode =
                        nextNode;
                }

                None => {

                    break;
                }
            }
        }


        leaves
    }


    // ---------------------------------------------------------
    // Count Keys
    // ---------------------------------------------------------

    pub fn countKeys(&self) -> usize {

        let mut count = 0;


        let leaves =
            self.getAllLeaves();


        for leafIndex in leaves {

            count +=
                self.nodes[leafIndex]
                    .keys
                    .len();
        }


        count
    }


    // ---------------------------------------------------------
    // Basic Tree Validation
    // ---------------------------------------------------------

    pub fn validate(&self) -> bool {

        // Root should not have a parent
        if self.nodes[self.root]
            .parent_node
            .is_some()
        {

            println!(
                "Root should not have a parent."
            );

            return false;
        }


        let mut queue =
            vec![self.root];


        while !queue.is_empty() {

            let nodeIndex =
                queue.remove(0);


            let node =
                &self.nodes[nodeIndex];


            if !node.is_leaf {

                // Internal nodes should have
                // one more child than keys
                if node.children.len()
                    != node.keys.len() + 1
                {

                    println!(
                        "Internal node {:?} has incorrect child count.",
                        node.keys
                    );

                    return false;
                }


                for childIndex in &node.children {

                    // Check parent connection
                    if self.nodes[*childIndex]
                        .parent_node
                        != Some(nodeIndex)
                    {

                        println!(
                            "Child {:?} has wrong parent.",
                            self.nodes[*childIndex].keys
                        );

                        return false;
                    }


                    queue.push(
                        *childIndex
                    );
                }
            }
        }


        true
    }


    // ---------------------------------------------------------
    // Verify Tree
    // ---------------------------------------------------------

    pub fn verifyTree(&self) -> bool {

        let mut leafDepth: Option<usize> =
            None;


        let mut leafKeys: Vec<i64> =
            Vec::new();


        let valid =
            self.verifyNode(
                self.root,
                None,
                None,
                0,
                &mut leafDepth,
                &mut leafKeys
            );


        if !valid {

            return false;
        }


        // Check global leaf order
        let mut sortedLeafKeys =
            leafKeys.clone();


        sortedLeafKeys.sort();


        if leafKeys != sortedLeafKeys {

            println!(
                "Leaf chain is not globally sorted."
            );

            return false;
        }


        println!(
            "Tree structure is valid."
        );


        true
    }


    // ---------------------------------------------------------
    // Verify Node
    // ---------------------------------------------------------

    fn verifyNode(
        &self,
        nodeIndex: usize,
        lowerBound: Option<i64>,
        upperBound: Option<i64>,
        depth: usize,
        leafDepth: &mut Option<usize>,
        leafKeys: &mut Vec<i64>
    ) -> bool {

        let node =
            &self.nodes[nodeIndex];


        // -----------------------------------------------------
        // Keys must be sorted
        // -----------------------------------------------------

        let mut sortedKeys =
            node.keys.clone();


        sortedKeys.sort();


        if node.keys != sortedKeys {

            println!(
                "Keys are not sorted in node: {:?}",
                node.keys
            );

            return false;
        }


        // -----------------------------------------------------
        // Check key bounds
        // -----------------------------------------------------

        for key in &node.keys {

            if let Some(lower) =
                lowerBound
            {

                if *key < lower {

                    println!(
                        "Key violates lower bound: {}",
                        key
                    );

                    return false;
                }
            }


            if let Some(upper) =
                upperBound
            {

                if *key > upper {

                    println!(
                        "Key violates upper bound: {}",
                        key
                    );

                    return false;
                }
            }
        }


        // -----------------------------------------------------
        // Maximum Occupancy
        // -----------------------------------------------------

        if node.keys.len()
            > self.maxKeys()
        {

            println!(
                "Node {:?} exceeds maxKeys={}",
                node.keys,
                self.maxKeys()
            );

            return false;
        }


        // -----------------------------------------------------
        // Minimum Occupancy
        // Root is exempt
        // -----------------------------------------------------

        if nodeIndex != self.root {

            if node.is_leaf
                && node.keys.len()
                    < self.minLeafKeys()
            {

                println!(
                    "Leaf {:?} below minLeafKeys={}",
                    node.keys,
                    self.minLeafKeys()
                );

                return false;
            }


            if !node.is_leaf
                && node.keys.len()
                    < self.minInternalKeys()
            {

                println!(
                    "Internal {:?} below minInternalKeys={}",
                    node.keys,
                    self.minInternalKeys()
                );

                return false;
            }
        }


        // -----------------------------------------------------
        // Leaf Node
        // -----------------------------------------------------

        if node.is_leaf {

            match *leafDepth {

                None => {

                    *leafDepth =
                        Some(depth);
                }


                Some(expectedDepth) => {

                    if expectedDepth
                        != depth
                    {

                        println!(
                            "Leaves are not at the same depth."
                        );

                        return false;
                    }
                }
            }


            for key in &node.keys {

                leafKeys.push(
                    *key
                );
            }


            return true;
        }


        // -----------------------------------------------------
        // Internal Node Child Count
        // -----------------------------------------------------

        if node.children.len()
            != node.keys.len() + 1
        {

            println!(
                "Internal node child count is incorrect."
            );

            return false;
        }


        // -----------------------------------------------------
        // Verify Children
        // -----------------------------------------------------

        for i in 0..node.children.len() {

            let childIndex =
                node.children[i];


            let childLower =
                if i > 0 {

                    Some(
                        node.keys[i - 1]
                    )
                }
                else {

                    lowerBound
                };


            let childUpper =
                if i < node.keys.len() {

                    Some(
                        node.keys[i]
                    )
                }
                else {

                    upperBound
                };


            let valid =
                self.verifyNode(
                    childIndex,
                    childLower,
                    childUpper,
                    depth + 1,
                    leafDepth,
                    leafKeys
                );


            if !valid {

                return false;
            }
        }


        true
    }
}