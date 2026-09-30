use ark_bn254::{
    Bn254,
    Config,
    Fq12,
    G1Affine,
    G2Affine,
};

use ark_ec::{
    bn::{
        BnConfig,
        G2Prepared,
    },
    pairing::Pairing,
    AffineRepr,
};

use ark_ff::{
    Field,
    One,
};

fn apply_line(
    f: &mut Fq12,
    coeff: &(ark_bn254::Fq2, ark_bn254::Fq2, ark_bn254::Fq2),
    p: &G1Affine,
) {
    let mut c0 = coeff.0;
    let mut c3 = coeff.1;
    let c4 = coeff.2;

    // BN254 is D-twist:
    //
    // c0 *= y_P
    // c3 *= x_P
    // mul_by_034(c0,c3,c4)

    c0.mul_assign_by_fp(&p.y);
    c3.mul_assign_by_fp(&p.x);

    f.mul_by_034(
        &c0,
        &c3,
        &c4,
    );
}

fn main() {
    println!(
        "=============================================="
    );

    println!(
        " BN254 Miller Loop Step Trace"
    );

    println!(
        "=============================================="
    );

    println!();

    let p =
        G1Affine::generator();

    let q =
        G2Affine::generator();

    let prepared_q =
        G2Prepared::<Config>::from(q);

    println!(
        "ATE loop count length : {}",
        Config::ATE_LOOP_COUNT.len()
    );

    println!(
        "Line coefficients     : {}",
        prepared_q.ell_coeffs.len()
    );

    println!();

    println!(
        "ATE_LOOP_COUNT:"
    );

    println!(
        "{:?}",
        Config::ATE_LOOP_COUNT
    );

    println!();

    let mut f =
        Fq12::one();

    let mut coeff_index =
        0usize;

    let mut step =
        0usize;

    println!(
        "Initial accumulator:"
    );

    println!(
        "f_0 = {:?}",
        f
    );

    println!();

    // This mirrors arkworks multi_miller_loop exactly.
    //
    // for i in (1..ATE_LOOP_COUNT.len()).rev()
    //
    // square except at first iteration
    // consume doubling line
    // possibly consume addition line

    for i in
        (1..Config::ATE_LOOP_COUNT.len()).rev()
    {
        println!(
            "=============================================="
        );

        println!(
            " Ate iteration i = {}",
            i
        );

        println!(
            "=============================================="
        );

        println!(
            "digit = {}",
            Config::ATE_LOOP_COUNT[i - 1]
        );

        println!();

        if i
            != Config::ATE_LOOP_COUNT.len() - 1
        {
            f.square_in_place();

            println!(
                "After square:"
            );

            println!(
                "{:?}",
                f
            );

            println!();
        } else {
            println!(
                "No square on first Ate iteration."
            );

            println!();
        }

        // ----------------------------------------------------
        // Doubling line
        // ----------------------------------------------------

        let coeff =
            prepared_q.ell_coeffs[
                coeff_index
            ];

        println!(
            "Consuming coeff[{}] as DOUBLING line",
            coeff_index
        );

        apply_line(
            &mut f,
            &coeff,
            &p,
        );

        coeff_index += 1;
        step += 1;

        println!(
            "Accumulator after line {}:",
            step
        );

        println!(
            "{:?}",
            f
        );

        println!();

        // ----------------------------------------------------
        // Optional addition line
        // ----------------------------------------------------

        let bit =
            Config::ATE_LOOP_COUNT[
                i - 1
            ];

        if bit == 1 || bit == -1 {
            let coeff =
                prepared_q.ell_coeffs[
                    coeff_index
                ];

            println!(
                "Consuming coeff[{}] as ADDITION line (digit {})",
                coeff_index,
                bit
            );

            apply_line(
                &mut f,
                &coeff,
                &p,
            );

            coeff_index += 1;
            step += 1;

            println!(
                "Accumulator after line {}:",
                step
            );

            println!(
                "{:?}",
                f
            );

            println!();
        }

        // We only need the beginning right now.
        if step >= 4 {
            break;
        }
    }

    println!(
        "=============================================="
    );

    println!(
        " Summary"
    );

    println!(
        "=============================================="
    );

    println!();

    println!(
        "Consumed coefficients : {}",
        coeff_index
    );

    println!(
        "Accumulator line steps: {}",
        step
    );
}