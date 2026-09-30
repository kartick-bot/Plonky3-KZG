use std::time::Instant;

use ark_bn254::{
    Config,
    Fq,
    Fq2,
    Fq12,
    G1Affine,
    G2Affine,
};
use ark_ec::{
    bn::{BnConfig, G2Prepared},
    AffineRepr,
};
use ark_ff::{
    BigInteger,
    CyclotomicMultSubgroup,
    Field,
    One,
    PrimeField,
};
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_challenger::{HashChallenger, SerializingChallenger64};
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::{
    extension::BinomialExtensionField,
    PrimeCharacteristicRing,
};
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_goldilocks::Goldilocks;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_sha256::Sha256;
use p3_symmetric::{
    CompressionFunctionFromHasher,
    SerializingHasher,
};
use p3_uni_stark::{prove, verify, StarkConfig};

type Val = Goldilocks;
type Challenge = BinomialExtensionField<Val, 2>;

// ============================================================
// Representation
// ============================================================

const LIMBS: usize = 16;
const LIMB_BITS: usize = 16;

const FQ_COUNT_IN_FQ12: usize = 12;
const FQ12_LIMBS: usize =
    FQ_COUNT_IN_FQ12 * LIMBS;

// ============================================================
// Operation selectors
//
// Exactly one selector is active on every non-padding row.
//
// INIT:
//      f = 1
//
// SQUARE:
//      next accumulator corresponds to f^2
//
// DOUBLE_LINE:
//      consume one G2 doubling line coefficient
//
// ADD_LINE:
//      consume a +/-Q addition-line coefficient
//
// FINAL_LINE:
//      the two Frobenius-related final lines
//
// PAD:
//      power-of-two trace padding
// ============================================================

const SEL_INIT: usize = 0;
const SEL_SQUARE: usize = 1;
const SEL_DOUBLE: usize = 2;
const SEL_ADD: usize = 3;
const SEL_FINAL: usize = 4;
const SEL_PAD: usize = 5;

const NUM_SELECTORS: usize = 6;

// ============================================================
// Metadata columns
// ============================================================

const COEFF_INDEX: usize =
    NUM_SELECTORS;

const ATE_INDEX: usize =
    COEFF_INDEX + 1;

const ATE_DIGIT: usize =
    ATE_INDEX + 1;

// ============================================================
// Current Fq12 accumulator.
//
// Layout:
// c0.c0.c0
// c0.c0.c1
// c0.c1.c0
// c0.c1.c1
// c0.c2.c0
// c0.c2.c1
// c1.c0.c0
// c1.c0.c1
// c1.c1.c0
// c1.c1.c1
// c1.c2.c0
// c1.c2.c1
//
// each Fq = 16 x 16-bit limbs
// ============================================================

const ACC_START: usize =
    ATE_DIGIT + 1;

const ACC_END: usize =
    ACC_START + FQ12_LIMBS;

// ============================================================
// Line coefficients after evaluation at P.
//
// For BN254 D-twist:
//
// c0 = ell_0 * y_P
// c3 = ell_1 * x_P
// c4 = ell_2
//
// 3 Fq2 values
// = 6 Fq values
// = 96 limbs
// ============================================================

const LINE_FQ_COUNT: usize = 6;

const LINE_START: usize =
    ACC_END;

const LINE_END: usize =
    LINE_START
    + LINE_FQ_COUNT * LIMBS;

// ============================================================
// Width
// ============================================================

const WIDTH: usize =
    LINE_END;

// ============================================================
// Operation enum
// ============================================================

#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
)]
enum Op {
    Init,
    Square,
    DoubleLine,
    AddLine,
    FinalLine,
    Pad,
}

// ============================================================
// Native schedule row
// ============================================================

#[derive(Clone)]
struct NativeRow {
    op: Op,

    coeff_index: usize,

    ate_index: usize,

    ate_digit: i8,

    accumulator: Fq12,

    // evaluated sparse line:
    //
    // (c0,c3,c4)
    line: Option<(Fq2, Fq2, Fq2)>,
}

// ============================================================
// Evaluate one BN254 D-twist line at G1 point P
// ============================================================

fn evaluated_line(
    coeff: &(Fq2, Fq2, Fq2),
    p: &G1Affine,
) -> (Fq2, Fq2, Fq2) {
    let mut c0 =
        coeff.0;

    let mut c3 =
        coeff.1;

    let c4 =
        coeff.2;

    c0.mul_assign_by_fp(
        &p.y
    );

    c3.mul_assign_by_fp(
        &p.x
    );

    (
        c0,
        c3,
        c4,
    )
}

// ============================================================
// Build complete Miller-loop execution schedule
// ============================================================

fn build_miller_schedule()
    -> Vec<NativeRow>
{
    let p =
        G1Affine::generator();

    let q =
        G2Affine::generator();

    let prepared_q =
        G2Prepared::<Config>::from(
            q
        );

    println!(
        "ATE loop digits       : {}",
        Config::ATE_LOOP_COUNT.len()
    );

    println!(
        "Prepared line coeffs  : {}",
        prepared_q.ell_coeffs.len()
    );

    let mut rows =
        Vec::<NativeRow>::new();

    let mut f =
        Fq12::one();

    let mut coeff_index =
        0usize;

    // ========================================================
    // Initial state
    // ========================================================

    rows.push(
        NativeRow {
            op: Op::Init,

            coeff_index: 0,

            ate_index:
                Config::ATE_LOOP_COUNT.len(),

            ate_digit: 0,

            accumulator: f,

            line: None,
        }
    );

    // ========================================================
    // Exact arkworks BN Miller loop structure
    //
    // for i in (1..ATE_LOOP_COUNT.len()).rev()
    //
    // square except first iteration
    // doubling line
    // optional +/- addition line
    // ========================================================

    for i in
        (1..Config::ATE_LOOP_COUNT.len())
            .rev()
    {
        let digit =
            Config::ATE_LOOP_COUNT[
                i - 1
            ];

        // ----------------------------------------------------
        // Square, except on very first iteration
        // ----------------------------------------------------

        if i
            != Config::ATE_LOOP_COUNT.len()
                - 1
        {
            f.square_in_place();

            rows.push(
                NativeRow {
                    op: Op::Square,

                    coeff_index,

                    ate_index: i,

                    ate_digit: digit,

                    accumulator: f,

                    line: None,
                }
            );
        }

        // ----------------------------------------------------
        // Doubling line
        // ----------------------------------------------------

        let coeff =
            prepared_q.ell_coeffs[
                coeff_index
            ];

        let line =
            evaluated_line(
                &coeff,
                &p,
            );

        f.mul_by_034(
            &line.0,
            &line.1,
            &line.2,
        );

        rows.push(
            NativeRow {
                op:
                    Op::DoubleLine,

                coeff_index,

                ate_index: i,

                ate_digit: digit,

                accumulator: f,

                line: Some(line),
            }
        );

        coeff_index += 1;

        // ----------------------------------------------------
        // Optional addition line
        // ----------------------------------------------------

        if digit == 1
            || digit == -1
        {
            let coeff =
                prepared_q.ell_coeffs[
                    coeff_index
                ];

            let line =
                evaluated_line(
                    &coeff,
                    &p,
                );

            f.mul_by_034(
                &line.0,
                &line.1,
                &line.2,
            );

            rows.push(
                NativeRow {
                    op:
                        Op::AddLine,

                    coeff_index,

                    ate_index: i,

                    ate_digit: digit,

                    accumulator: f,

                    line:
                        Some(line),
                }
            );

            coeff_index += 1;
        }
    }

    // ========================================================
    // BN negative-X handling
    //
    // BN254's Config tells us whether this is needed.
    //
    // For the moment we perform it natively in the witness.
    // It will later become a constrained operation row.
    // ========================================================

    if Config::X_IS_NEGATIVE {
        f.cyclotomic_inverse_in_place();

        // We deliberately do not add a separate operation type
        // yet. BN254's actual config determines whether this
        // branch occurs.
    }

    // ========================================================
    // Two final Frobenius-related lines.
    // Arkworks consumes exactly two more prepared coefficients.
    // ========================================================

    for _ in 0..2 {
        let coeff =
            prepared_q.ell_coeffs[
                coeff_index
            ];

        let line =
            evaluated_line(
                &coeff,
                &p,
            );

        f.mul_by_034(
            &line.0,
            &line.1,
            &line.2,
        );

        rows.push(
            NativeRow {
                op:
                    Op::FinalLine,

                coeff_index,

                ate_index: 0,

                ate_digit: 0,

                accumulator: f,

                line:
                    Some(line),
            }
        );

        coeff_index += 1;
    }

    assert_eq!(
        coeff_index,
        prepared_q
            .ell_coeffs
            .len(),
        "not all prepared Miller coefficients were consumed"
    );

    println!(
        "Consumed coefficients : {}",
        coeff_index
    );

    println!(
        "Unpadded schedule rows : {}",
        rows.len()
    );

    rows
}

// ============================================================
// Fq -> 16-bit limbs
// ============================================================

fn fq_to_limbs(
    x: &Fq,
) -> [u64; LIMBS] {
    let bigint =
        x.into_bigint();

    let bytes =
        bigint.to_bytes_le();

    let mut limbs =
        [0u64; LIMBS];

    for i in 0..LIMBS {
        let lo_index =
            2 * i;

        let hi_index =
            lo_index + 1;

        let lo =
            if lo_index
                < bytes.len()
            {
                bytes[lo_index]
                    as u64
            } else {
                0
            };

        let hi =
            if hi_index
                < bytes.len()
            {
                bytes[hi_index]
                    as u64
            } else {
                0
            };

        limbs[i] =
            lo | (hi << 8);
    }

    limbs
}

// ============================================================
// Push Fq as limbs
// ============================================================

fn push_fq(
    row: &mut Vec<Val>,
    x: &Fq,
) {
    for limb in
        fq_to_limbs(x)
    {
        row.push(
            Val::from_u64(
                limb
            )
        );
    }
}

// ============================================================
// Push Fq2
// ============================================================

fn push_fq2(
    row: &mut Vec<Val>,
    x: &Fq2,
) {
    push_fq(
        row,
        &x.c0,
    );

    push_fq(
        row,
        &x.c1,
    );
}

// ============================================================
// Push Fq12 in exactly 12-Fq order
// ============================================================

fn push_fq12(
    row: &mut Vec<Val>,
    x: &Fq12,
) {
    // c0.c0
    push_fq2(
        row,
        &x.c0.c0,
    );

    // c0.c1
    push_fq2(
        row,
        &x.c0.c1,
    );

    // c0.c2
    push_fq2(
        row,
        &x.c0.c2,
    );

    // c1.c0
    push_fq2(
        row,
        &x.c1.c0,
    );

    // c1.c1
    push_fq2(
        row,
        &x.c1.c1,
    );

    // c1.c2
    push_fq2(
        row,
        &x.c1.c2,
    );
}

// ============================================================
// Encode operation selectors
// ============================================================

fn push_selectors(
    row: &mut Vec<Val>,
    op: Op,
) {
    let active =
        match op {
            Op::Init =>
                SEL_INIT,

            Op::Square =>
                SEL_SQUARE,

            Op::DoubleLine =>
                SEL_DOUBLE,

            Op::AddLine =>
                SEL_ADD,

            Op::FinalLine =>
                SEL_FINAL,

            Op::Pad =>
                SEL_PAD,
        };

    for i in 0..NUM_SELECTORS {
        row.push(
            if i == active {
                Val::ONE
            } else {
                Val::ZERO
            }
        );
    }
}

// ============================================================
// Convert one NativeRow into trace row
// ============================================================

fn encode_row(
    native: &NativeRow,
) -> Vec<Val> {
    let mut row =
        Vec::with_capacity(
            WIDTH
        );

    push_selectors(
        &mut row,
        native.op,
    );

    row.push(
        Val::from_u64(
            native.coeff_index
                as u64
        )
    );

    row.push(
        Val::from_u64(
            native.ate_index
                as u64
        )
    );

    // Encode digit:
    //
    // -1 -> Goldilocks(-1)
    //  0 -> 0
    // +1 -> 1

    let digit_val =
        match native.ate_digit {
            -1 =>
                -Val::ONE,

            0 =>
                Val::ZERO,

            1 =>
                Val::ONE,

            x =>
                panic!(
                    "unexpected Ate digit {}",
                    x
                ),
        };

    row.push(
        digit_val
    );

    // --------------------------------------------------------
    // Current Fq12 accumulator
    // --------------------------------------------------------

    push_fq12(
        &mut row,
        &native.accumulator,
    );

    // --------------------------------------------------------
    // Evaluated line coefficients
    // --------------------------------------------------------

    match native.line {
        Some(
            (
                ref c0,
                ref c3,
                ref c4,
            )
        ) => {
            push_fq2(
                &mut row,
                c0,
            );

            push_fq2(
                &mut row,
                c3,
            );

            push_fq2(
                &mut row,
                c4,
            );
        }

        None => {
            for _ in
                0..(
                    LINE_FQ_COUNT
                    * LIMBS
                )
            {
                row.push(
                    Val::ZERO
                );
            }
        }
    }

    assert_eq!(
        row.len(),
        WIDTH
    );

    row
}

// ============================================================
// Next power of two
// ============================================================

fn next_power_of_two(
    n: usize,
) -> usize {
    n.next_power_of_two()
}

// ============================================================
// Trace construction
// ============================================================

fn make_trace()
    -> (
        RowMajorMatrix<Val>,
        usize,
        usize,
    )
{
    let schedule =
        build_miller_schedule();

    let real_rows =
        schedule.len();

    let height =
        next_power_of_two(
            real_rows
        );

    let mut values =
        Vec::with_capacity(
            WIDTH * height
        );

    // --------------------------------------------------------
    // Real pairing schedule
    // --------------------------------------------------------

    for native in &schedule {
        let row =
            encode_row(native);

        values.extend_from_slice(
            &row
        );
    }

    // --------------------------------------------------------
    // Padding
    //
    // Copy final accumulator into every padding row.
    // --------------------------------------------------------

    let final_schedule_row =
        schedule
            .last()
            .expect(
                "empty pairing schedule"
            );

    let final_accumulator =
        final_schedule_row
            .accumulator;

    // The last real line row stores the index of the coefficient
    // consumed on that row. Since all 87 prepared coefficients
    // have been consumed when we enter padding, the padding state
    // must carry the post-consumption index = last_index + 1.
    let final_coeff_index =
        final_schedule_row
            .coeff_index
            + 1;

    while
        values.len()
            / WIDTH
        < height
    {
        let pad =
            NativeRow {
                op: Op::Pad,

                coeff_index:
                    final_coeff_index,

                ate_index: 0,

                ate_digit: 0,

                accumulator:
                    final_accumulator,

                line: None,
            };

        let row =
            encode_row(
                &pad
            );

        values.extend_from_slice(
            &row
        );
    }

    (
        RowMajorMatrix::new(
            values,
            WIDTH,
        ),
        real_rows,
        height,
    )
}

// ============================================================
// AIR
// ============================================================

struct PairingScheduleAir;

impl<F> BaseAir<F>
    for PairingScheduleAir
{
    fn width(&self) -> usize {
        WIDTH
    }
}

impl<AB: AirBuilder> Air<AB>
    for PairingScheduleAir
{
    fn eval(
        &self,
        builder: &mut AB,
    ) {
        let main =
            builder.main();

        let local =
            main.current_slice();

        let next =
            main.next_slice();

        // ====================================================
        // Selector booleanity
        // ====================================================

        let mut selector_sum =
            AB::Expr::ZERO;

        for i in 0..NUM_SELECTORS {
            let s: AB::Expr =
                local[i].into();

            builder.assert_zero(
                s.clone()
                    * (
                        s.clone()
                        - AB::Expr::ONE
                    )
            );

            selector_sum =
                selector_sum + s;
        }

        // Exactly one operation per row.

        builder.assert_one(
            selector_sum
        );

        // ====================================================
        // First row must be INIT.
        // ====================================================

        builder
            .when_first_row()
            .assert_one(
                local[SEL_INIT]
            );

        // ====================================================
        // INIT accumulator = 1 in Fq12.
        //
        // First Fq limb = 1.
        // Every other limb = 0.
        // ====================================================

        builder
            .when_first_row()
            .assert_one(
                local[ACC_START]
            );

        for i in
            (ACC_START + 1)..ACC_END
        {
            builder
                .when_first_row()
                .assert_zero(
                    local[i]
                );
        }

        // INIT has no line coefficients.

        for i in
            LINE_START..LINE_END
        {
            builder
                .when_first_row()
                .assert_zero(
                    local[i]
                );
        }

        // ====================================================
        // Padding rows must remain padding.
        // ====================================================

        {
            let pad: AB::Expr =
                local[
                    SEL_PAD
                ]
                .into();

            let next_pad: AB::Expr =
                next[
                    SEL_PAD
                ]
                .into();

            builder
                .when_transition()
                .when(
                    pad
                )
                .assert_one(
                    next_pad
                );
        }

        // ====================================================
        // Padding accumulator is constant.
        // ====================================================

        for i in
            ACC_START..ACC_END
        {
            let local_acc: AB::Expr =
                local[i]
                    .into();

            let next_acc: AB::Expr =
                next[i]
                    .into();

            let pad: AB::Expr =
                local[
                    SEL_PAD
                ]
                .into();

            builder
                .when_transition()
                .when(
                    pad
                )
                .assert_eq(
                    local_acc,
                    next_acc,
                );
        }

        // ====================================================
        // Padding coefficient index is constant.
        //
        // The first padding row carries the post-consumption
        // coefficient index (87 for BN254 here), and every
        // subsequent padding row must preserve it.
        // ====================================================

        {
            let pad: AB::Expr =
                local[
                    SEL_PAD
                ]
                .into();

            let coeff: AB::Expr =
                local[
                    COEFF_INDEX
                ]
                .into();

            let next_coeff: AB::Expr =
                next[
                    COEFF_INDEX
                ]
                .into();

            builder
                .when_transition()
                .when(
                    pad
                )
                .assert_eq(
                    coeff,
                    next_coeff,
                );
        }

        // ====================================================
        // Padding rows have no line.
        // ====================================================

        for i in
            LINE_START..LINE_END
        {
            let pad: AB::Expr =
                local[
                    SEL_PAD
                ]
                .into();

            builder
                .when(
                    pad
                )
                .assert_zero(
                    local[i]
                );
        }

        // ====================================================
        // Square rows do not consume a line.
        // ====================================================

        for i in
            LINE_START..LINE_END
        {
            let square: AB::Expr =
                local[
                    SEL_SQUARE
                ]
                .into();

            builder
                .when(
                    square
                )
                .assert_zero(
                    local[i]
                );
        }

        // ====================================================
        // Coefficient index transition skeleton.
        //
        // Square:
        //     coeff_next = coeff
        //
        // Double/Add/Final line:
        //     coeff_next = coeff+1
        //
        // Note:
        // The accumulator arithmetic itself is the NEXT
        // layer we plug into these rows.
        // ====================================================

        let coeff: AB::Expr =
            local[
                COEFF_INDEX
            ]
            .into();

        let next_coeff: AB::Expr =
            next[
                COEFF_INDEX
            ]
            .into();

        let square: AB::Expr =
            local[
                SEL_SQUARE
            ]
            .into();

        builder
            .when_transition()
            .when(
                square
            )
            .assert_eq(
                next_coeff.clone(),
                coeff.clone(),
            );

        let dbl: AB::Expr =
            local[
                SEL_DOUBLE
            ]
            .into();

        let add: AB::Expr =
            local[
                SEL_ADD
            ]
            .into();

        let final_line: AB::Expr =
            local[
                SEL_FINAL
            ]
            .into();

        let consume =
            dbl
            + add
            + final_line;

        builder
            .when_transition()
            .when(
                consume
            )
            .assert_eq(
                next_coeff,
                coeff
                    + AB::Expr::ONE,
            );

        // ====================================================
        // Ate digit is restricted to {-1,0,+1}.
        //
        // x(x-1)(x+1)=0
        //
        // degree 3.
        // ====================================================

        let digit: AB::Expr =
            local[
                ATE_DIGIT
            ]
            .into();

        builder.assert_zero(
            digit.clone()
                * (
                    digit.clone()
                    - AB::Expr::ONE
                )
                * (
                    digit
                    + AB::Expr::ONE
                )
        );

        // ====================================================
        // IMPORTANT:
        //
        // We intentionally DO NOT yet constrain:
        //
        // next_f = f^2
        //
        // or
        //
        // next_f =
        // f * mul_by_034(line)
        //
        // Those are exactly the non-native Fq12 constraints
        // we add into these same transition rows next.
        // ====================================================
    }
}

// ============================================================
// Main
// ============================================================

fn main() {
    println!(
        "=============================================="
    );

    println!(
        " Plonky3 BN254 Pairing STARK"
    );

    println!(
        " Row-Based Miller Loop Skeleton"
    );

    println!(
        "=============================================="
    );

    println!();

    println!(
        "Native proving field : Goldilocks"
    );

    println!(
        "Pairing curve        : BN254"
    );

    println!(
        "Accumulator field    : Fq12"
    );

    println!(
        "Fq representation    : 16 x 16-bit limbs"
    );

    println!();

    let (
        trace,
        real_rows,
        trace_height,
    ) =
        make_trace();

    println!();

    println!(
        "=============================================="
    );

    println!(
        " Trace layout"
    );

    println!(
        "=============================================="
    );

    println!();

    println!(
        "Selectors             : {}",
        NUM_SELECTORS
    );

    println!(
        "Metadata columns      : 3"
    );

    println!(
        "Fq12 accumulator      : {} limbs",
        FQ12_LIMBS
    );

    println!(
        "Line coefficients     : {} limbs",
        LINE_FQ_COUNT
            * LIMBS
    );

    println!(
        "Trace width           : {} columns",
        WIDTH
    );

    println!(
        "Real schedule rows    : {}",
        real_rows
    );

    println!(
        "Padded trace height   : {}",
        trace_height
    );

    println!(
        "Trace cells           : {}",
        WIDTH
            * trace_height
    );

    println!();

    // ========================================================
    // Plonky3 config
    // ========================================================

    type ByteHash =
        Sha256;

    type FieldHash =
        SerializingHasher<
            ByteHash
        >;

    let byte_hash =
        ByteHash {};

    let field_hash =
        FieldHash::new(
            byte_hash
        );

    type Compress =
        CompressionFunctionFromHasher<
            ByteHash,
            2,
            32,
        >;

    let compress =
        Compress::new(
            byte_hash
        );

    type ValMmcs =
        MerkleTreeMmcs<
            Val,
            u8,
            FieldHash,
            Compress,
            2,
            32,
        >;

    let val_mmcs =
        ValMmcs::new(
            field_hash,
            compress,
            3,
        );

    type ChallengeMmcs =
        ExtensionMmcs<
            Val,
            Challenge,
            ValMmcs,
        >;

    let challenge_mmcs =
        ChallengeMmcs::new(
            val_mmcs.clone()
        );

    type Dft =
        Radix2DitParallel<Val>;

    let dft =
        Dft::default();

    let fri_params =
        FriParameters::new_benchmark(
            challenge_mmcs
        );

    type Pcs =
        TwoAdicFriPcs<
            Val,
            Dft,
            ValMmcs,
            ChallengeMmcs,
        >;

    let pcs =
        Pcs::new(
            dft,
            val_mmcs,
            fri_params,
        );

    type Challenger =
        SerializingChallenger64<
            Val,
            HashChallenger<
                u8,
                ByteHash,
                32,
            >,
        >;

    let challenger =
        Challenger::from_hasher(
            vec![],
            byte_hash,
        );

    type ConfigType =
        StarkConfig<
            Pcs,
            Challenge,
            Challenger,
        >;

    let config =
        ConfigType::new(
            pcs,
            challenger,
        );

    // ========================================================
    // Prove
    // ========================================================

    let prove_start =
        Instant::now();

    let proof =
        prove(
            &config,
            &PairingScheduleAir,
            trace,
            &[],
        )
        .expect(
            "proof generation failed"
        );

    let proving_time =
        prove_start.elapsed();

    let proof_bytes =
        postcard::to_allocvec(
            &proof
        )
        .expect(
            "proof serialization failed"
        );

    // ========================================================
    // Verify
    // ========================================================

    let verify_start =
        Instant::now();

    verify(
        &config,
        &PairingScheduleAir,
        &proof,
        &[],
    )
    .expect(
        "verification failed"
    );

    let verification_time =
        verify_start.elapsed();

    println!(
        "=============================================="
    );

    println!(
        " STARK result"
    );

    println!(
        "=============================================="
    );

    println!();

    println!(
        "Proof size        : {} bytes",
        proof_bytes.len()
    );

    println!(
        "Proving time      : {:?}",
        proving_time
    );

    println!(
        "Verification time : {:?}",
        verification_time
    );

    println!();

    println!(
        "Full BN254 Miller schedule trace verified!"
    );

    println!();

    println!(
        "NOTE:"
    );

    println!(
        "This proof currently constrains the full pairing"
    );

    println!(
        "schedule/state layout, selectors and coefficient"
    );

    println!(
        "consumption. Fq12 square/mul_by_034 arithmetic"
    );

    println!(
        "constraints are the next layer."
    );
}