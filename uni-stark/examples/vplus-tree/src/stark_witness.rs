
 //! Data interface for an existing V+ Tree KZG opening.
 //! This structure is not itself a STARK proof or AIR constraint.

use ark_bn254::{Fr, G1Affine, G2Affine};

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
