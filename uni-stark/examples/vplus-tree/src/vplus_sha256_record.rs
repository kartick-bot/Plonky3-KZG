//! Exact SHA-256 record-message adapter for V+ Tree.
//!
//! This is INPUT PREPARATION for a future SHA-256 AIR, not an AIR itself.
//! The message is exactly the one used by leaf_kzg::internalRecordEvaluation:
//!     "VPLUS_INTERNAL_NODE" || i64::to_be_bytes(key) ||
//!     compressed_G1(LPC) || compressed_G1(RPC).
//!
//! For BN254 compressed affine G1 points, the input is 91 bytes and pads
//! to two SHA-256 64-byte blocks. `blocks` below are the actual SHA-256
//! compression inputs, with the conventional initial chaining state.

use ark_bn254::{Fr, G1Affine};
use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;
use sha2::{Digest, Sha256};

pub const VPLUS_INTERNAL_DOMAIN: &[u8; 19] = b"VPLUS_INTERNAL_NODE";
pub const VPLUS_RECORD_MESSAGE_SIZE: usize = 19 + 8 + 32 + 32;
pub const SHA256_PADDED_SIZE: usize = 128;

pub const SHA256_IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

#[derive(Clone, Debug)]
pub struct InternalRecordHashInput {
    pub message: [u8; VPLUS_RECORD_MESSAGE_SIZE],
    pub blocks: [[u8; 64]; 2],
    pub digest: [u8; 32],
    pub evaluation_value: Fr,
}

pub fn make_internal_record_hash_input(
    key: i64,
    left: &G1Affine,
    right: &G1Affine,
) -> InternalRecordHashInput {
    let mut left_bytes = Vec::new();
    let mut right_bytes = Vec::new();
    left.serialize_compressed(&mut left_bytes).expect("LPC serialization failed");
    right.serialize_compressed(&mut right_bytes).expect("RPC serialization failed");
    assert_eq!(left_bytes.len(), 32, "BN254 compressed LPC must be 32 bytes");
    assert_eq!(right_bytes.len(), 32, "BN254 compressed RPC must be 32 bytes");

    let mut message = [0u8; VPLUS_RECORD_MESSAGE_SIZE];
    message[..19].copy_from_slice(VPLUS_INTERNAL_DOMAIN);
    message[19..27].copy_from_slice(&key.to_be_bytes());
    message[27..59].copy_from_slice(&left_bytes);
    message[59..91].copy_from_slice(&right_bytes);
    let actual_message_len = VPLUS_RECORD_MESSAGE_SIZE;

    let mut padded = [0u8; SHA256_PADDED_SIZE];
    padded[..actual_message_len].copy_from_slice(&message[..actual_message_len]);
    padded[actual_message_len] = 0x80;
    let bit_len = (actual_message_len as u64) * 8;
    padded[120..128].copy_from_slice(&bit_len.to_be_bytes());
    let blocks: [[u8; 64]; 2] = [
        padded[0..64].try_into().unwrap(),
        padded[64..128].try_into().unwrap(),
    ];

    let hash = Sha256::digest(&message[..actual_message_len]);
    let mut digest = [0u8; 32];
    digest.copy_from_slice(&hash);

    let high = Fr::from_be_bytes_mod_order(&digest[..16]);
    let low = Fr::from_be_bytes_mod_order(&digest[16..]);

    InternalRecordHashInput {
        message,
        blocks,
        digest,
        evaluation_value: high + low,
    }
}


/// SHA-256 round constants, FIPS 180-4 section 4.2.2.
const SHA256_K: [u32; 64] = [
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
];

/// The complete witness of one SHA-256 compression block.
/// A future AIR must constrain each schedule word and state transition;
/// storing these rows by itself does NOT prove their correctness.
#[derive(Clone, Debug)]
pub struct Sha256Round {
    pub block_index: usize,
    pub round_index: usize,
    pub state_before: [u32; 8],
    pub schedule_word: u32,
    pub temp1: u32,
    pub temp2: u32,
    pub state_after: [u32; 8],
}

#[derive(Clone, Debug)]
pub struct Sha256CompressionWitness {
    pub rounds: Vec<Sha256Round>,
    pub digest: [u8; 32],
}

/// Generate *all 128* round witness rows from exactly the two padded
/// blocks created by `make_internal_record_hash_input`.
pub fn make_sha256_compression_witness(
    input: &InternalRecordHashInput,
) -> Sha256CompressionWitness {
    let mut h = SHA256_IV;
    let mut rounds = Vec::with_capacity(128);

    for (block_index, block) in input.blocks.iter().enumerate() {
        let mut w = [0u32; 64];
        for t in 0..16 {
            w[t] = u32::from_be_bytes(block[4*t..4*t+4].try_into().unwrap());
        }
        for t in 16..64 {
            let s0 = w[t-15].rotate_right(7) ^ w[t-15].rotate_right(18) ^ (w[t-15] >> 3);
            let s1 = w[t-2].rotate_right(17) ^ w[t-2].rotate_right(19) ^ (w[t-2] >> 10);
            w[t] = w[t-16].wrapping_add(s0).wrapping_add(w[t-7]).wrapping_add(s1);
        }
        let mut v = h;
        for t in 0..64 {
            let before = v;
            let big_s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ ((!v[4]) & v[6]);
            let t1 = v[7].wrapping_add(big_s1).wrapping_add(ch)
                .wrapping_add(SHA256_K[t]).wrapping_add(w[t]);
            let big_s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = big_s0.wrapping_add(maj);
            v = [t1.wrapping_add(t2),v[0],v[1],v[2],v[3].wrapping_add(t1),v[4],v[5],v[6]];
            rounds.push(Sha256Round {
                block_index, round_index: t, state_before: before,
                schedule_word: w[t], temp1: t1, temp2: t2,
                state_after: v,
            });
        }
        for i in 0..8 { h[i] = h[i].wrapping_add(v[i]); }
    }
    let mut digest = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        digest[4*i..4*i+4].copy_from_slice(&word.to_be_bytes());
    }
    assert_eq!(digest, input.digest, "SHA-256 round trace differs from sha2::Sha256");
    Sha256CompressionWitness { rounds, digest }
}

/// Compare this adapter to the original V+ Tree hash implementation.
/// Invoke during development, before building the SHA-256 AIR trace.
pub fn check_against_existing_tree(
    key: i64,
    left: &G1Affine,
    right: &G1Affine,
) -> InternalRecordHashInput {
    let input = make_internal_record_hash_input(key, left, right);
    let rounds = make_sha256_compression_witness(&input);
    assert_eq!(rounds.rounds.len(), 128);
    assert_eq!(rounds.digest, input.digest);
    assert_eq!(
        input.evaluation_value,
        crate::leaf_kzg::internalRecordEvaluation(key, left, right),
        "SHA-256 input adapter differs from original V+ Tree record hash"
    );
    input
}