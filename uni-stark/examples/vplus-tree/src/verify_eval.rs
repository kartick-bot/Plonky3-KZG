use ark_bn254::{
    Bn254,
    Fr,
    G1Affine,
    G1Projective,
    G2Projective,
};

use ark_ec::{
    pairing::Pairing,
    CurveGroup,
};

use crate::leaf_kzg::KzgSrs;


// ---------------------------------------------------------
// Verify One KZG Evaluation
//
// Checks:
//
//     e(C - yG1, G2)
//         =
//     e(W, tauG2 - iG2)
//
// where:
//
//     C = commitment to f(X)
//     i = evaluation point
//     y = f(i)
//     W = KZG witness
// ---------------------------------------------------------

pub fn verifyEval(
    commitment: &G1Affine,
    i: Fr,
    y: Fr,
    witness: &G1Affine,
    srs: &KzgSrs
) -> bool {

    // -----------------------------------------------------
    // C - yG1
    // -----------------------------------------------------

    let left_g1 =
        (
            G1Projective::from(
                *commitment
            )
            -
            G1Projective::from(
                srs.g1_powers[0]
            )
            * y
        )
        .into_affine();


    // -----------------------------------------------------
    // tauG2 - iG2
    // -----------------------------------------------------

    let right_g2 =
        (
            G2Projective::from(
                srs.tau_g2
            )
            -
            G2Projective::from(
                srs.g2
            )
            * i
        )
        .into_affine();


    // -----------------------------------------------------
    // Pairing Check
    // -----------------------------------------------------

    let left_pairing =
        Bn254::pairing(
            left_g1,
            srs.g2
        );


    let right_pairing =
        Bn254::pairing(
            *witness,
            right_g2
        );


    left_pairing
        ==
    right_pairing
}