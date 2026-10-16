use ark_bn254::{Bn254, Config, Fq, Fq2, Fq6, Fq12, Fr, G1Affine, G2Affine};
use ark_ec::{
    AffineRepr,
    bn::{BnConfig, G2Prepared},
    pairing::Pairing,
};
use ark_ff::{BigInteger, CyclotomicMultSubgroup, Field, One, PrimeField, Zero};
use num_bigint::BigUint;

fn fq12_to_coords(x: &Fq12) -> [Fq; 12] {
    [
        x.c0.c0.c0, x.c0.c0.c1, x.c0.c1.c0, x.c0.c1.c1, x.c0.c2.c0, x.c0.c2.c1, x.c1.c0.c0,
        x.c1.c0.c1, x.c1.c1.c0, x.c1.c1.c1, x.c1.c2.c0, x.c1.c2.c1,
    ]
}

fn coords_to_fq12(a: &[Fq; 12]) -> Fq12 {
    Fq12::new(
        Fq6::new(
            Fq2::new(a[0], a[1]),
            Fq2::new(a[2], a[3]),
            Fq2::new(a[4], a[5]),
        ),
        Fq6::new(
            Fq2::new(a[6], a[7]),
            Fq2::new(a[8], a[9]),
            Fq2::new(a[10], a[11]),
        ),
    )
}

fn basis_decode(index: usize) -> (usize, usize, usize) {
    let w = index / 6;
    let rem = index % 6;
    let v = rem / 2;
    let u = rem % 2;
    (u, v, w)
}

fn basis_encode(u: usize, v: usize, w: usize) -> usize {
    w * 6 + v * 2 + u
}

fn basis_product(i: usize, j: usize) -> Vec<(usize, i64)> {
    let (u1, v1, w1) = basis_decode(i);
    let (u2, v2, w2) = basis_decode(j);

    let mut coeff: i64 = 1;
    let mut u = u1 + u2;
    let mut v = v1 + v2;
    let mut w = w1 + w2;

    if u >= 2 {
        u -= 2;
        coeff = -coeff;
    }

    if w >= 2 {
        w -= 2;
        v += 1;
    }

    if v < 3 {
        return vec![(basis_encode(u, v, w), coeff)];
    }

    v -= 3;

    if u == 0 {
        vec![
            (basis_encode(0, v, w), 9 * coeff),
            (basis_encode(1, v, w), coeff),
        ]
    } else {
        vec![
            (basis_encode(0, v, w), -coeff),
            (basis_encode(1, v, w), 9 * coeff),
        ]
    }
}

fn add_scaled_fq(acc: &mut Fq, x: Fq, coeff: i64) {
    let mag = coeff.unsigned_abs();
    let scaled = x * Fq::from(mag);
    if coeff >= 0 {
        *acc += scaled;
    } else {
        *acc -= scaled;
    }
}

fn fq12_mul_direct(a: &Fq12, b: &Fq12) -> Fq12 {
    let aa = fq12_to_coords(a);
    let bb = fq12_to_coords(b);
    let mut out = [Fq::zero(); 12];

    for i in 0..12 {
        for j in 0..12 {
            let prod = aa[i] * bb[j];
            for (k, coeff) in basis_product(i, j) {
                add_scaled_fq(&mut out[k], prod, coeff);
            }
        }
    }

    coords_to_fq12(&out)
}

fn sparse_line_fq12(line: &(Fq2, Fq2, Fq2)) -> Fq12 {
    Fq12::new(
        Fq6::new(line.0, Fq2::zero(), Fq2::zero()),
        Fq6::new(line.1, line.2, Fq2::zero()),
    )
}

fn evaluated_line(coeff: &(Fq2, Fq2, Fq2), p: &G1Affine) -> (Fq2, Fq2, Fq2) {
    let mut c0 = coeff.0;
    let mut c3 = coeff.1;
    let c4 = coeff.2;

    c0.mul_assign_by_fp(&p.y);
    c3.mul_assign_by_fp(&p.x);

    (c0, c3, c4)
}

fn fq_big(x: &Fq) -> BigUint {
    BigUint::from_bytes_le(&x.into_bigint().to_bytes_le())
}

fn final_exponent() -> BigUint {
    let q = BigUint::from_bytes_le(&Fq::MODULUS.to_bytes_le());
    let r = BigUint::from_bytes_le(&Fr::MODULUS.to_bytes_le());
    (q.pow(12u32) - BigUint::from(1u64)) / r
}

fn build_miller() -> Fq12 {
    let p = G1Affine::generator();
    let q = G2Affine::generator();
    let prepared = G2Prepared::<Config>::from(q);

    let mut f = Fq12::one();
    let mut coeff_index = 0usize;

    for i in (1..Config::ATE_LOOP_COUNT.len()).rev() {
        if i != Config::ATE_LOOP_COUNT.len() - 1 {
            let mut ark = f;
            ark.square_in_place();

            let direct = fq12_mul_direct(&f, &f);
            assert_eq!(direct, ark, "square direct mismatch");

            f = ark;
        }

        let coeff = prepared.ell_coeffs[coeff_index];
        let line = evaluated_line(&coeff, &p);
        let sparse = sparse_line_fq12(&line);

        let direct = fq12_mul_direct(&f, &sparse);

        f.mul_by_034(&line.0, &line.1, &line.2);
        assert_eq!(direct, f, "mul_by_034 direct mismatch");

        coeff_index += 1;

        let digit = Config::ATE_LOOP_COUNT[i - 1];
        if digit == 1 || digit == -1 {
            let coeff = prepared.ell_coeffs[coeff_index];
            let line = evaluated_line(&coeff, &p);
            let sparse = sparse_line_fq12(&line);

            let direct = fq12_mul_direct(&f, &sparse);

            f.mul_by_034(&line.0, &line.1, &line.2);
            assert_eq!(direct, f, "addition mul_by_034 direct mismatch");

            coeff_index += 1;
        }
    }

    if Config::X_IS_NEGATIVE {
        f.cyclotomic_inverse_in_place();
    }

    for _ in 0..2 {
        let coeff = prepared.ell_coeffs[coeff_index];
        let line = evaluated_line(&coeff, &p);
        let sparse = sparse_line_fq12(&line);

        let direct = fq12_mul_direct(&f, &sparse);

        f.mul_by_034(&line.0, &line.1, &line.2);
        assert_eq!(direct, f, "final mul_by_034 direct mismatch");

        coeff_index += 1;
    }

    assert_eq!(coeff_index, prepared.ell_coeffs.len());
    f
}

fn main() {
    println!("==============================================");
    println!(" BN254 pairing diagnostic");
    println!("==============================================");

    let p = G1Affine::generator();
    let q = G2Affine::generator();

    let ours_miller = build_miller();
    let native_miller = Bn254::multi_miller_loop([p], [q]).0;

    println!("Config::X_IS_NEGATIVE : {}", Config::X_IS_NEGATIVE);
    println!("Miller matches arkworks: {}", ours_miller == native_miller);

    if ours_miller != native_miller {
        println!("STOP: Miller loop mismatch.");
        return;
    }

    let e = final_exponent();
    println!("Final exponent bits    : {}", e.bits());

    let direct_pairing = Bn254::pairing(p, q).0;

    // Arkworks generic pow using exponent limbs.
    let pow_limbs = e.to_u64_digits();
    let ark_pow = ours_miller.pow(pow_limbs.as_slice());

    println!(
        "Full exponent via arkworks pow matches pairing: {}",
        ark_pow == direct_pairing
    );

    if ark_pow != direct_pairing {
        println!("STOP: exponent formula itself is wrong.");
        return;
    }

    // Right-to-left exponentiation using ARKWORKS multiplication.
    let mut acc_ark = Fq12::one();
    let mut base_ark = ours_miller;
    let mut exp_ark = e.clone();

    let mut round = 0usize;
    while !exp_ark.is_zero() {
        if (&exp_ark & BigUint::from(1u64)) == BigUint::from(1u64) {
            acc_ark *= base_ark;
        }
        base_ark.square_in_place();
        exp_ark >>= 1usize;
        round += 1;
    }

    println!(
        "RTL exponent using arkworks mul matches pairing: {} ({} rounds)",
        acc_ark == direct_pairing,
        round
    );

    if acc_ark != direct_pairing {
        println!("STOP: right-to-left exponent state logic is wrong.");
        return;
    }

    // Right-to-left exponentiation using OUR generic direct multiply.
    let mut acc = Fq12::one();
    let mut base = ours_miller;
    let mut exp = e.clone();

    let mut round = 0usize;

    while !exp.is_zero() {
        if (&exp & BigUint::from(1u64)) == BigUint::from(1u64) {
            let ours = fq12_mul_direct(&acc, &base);
            let ark = acc * base;

            if ours != ark {
                println!("GENERIC MUL MISMATCH at acc multiply, round {}", round);
                println!("exp LSB = 1");
                return;
            }

            acc = ours;
        }

        let ours_sq = fq12_mul_direct(&base, &base);
        let mut ark_sq = base;
        ark_sq.square_in_place();

        if ours_sq != ark_sq {
            println!("GENERIC MUL MISMATCH at base square, round {}", round);
            return;
        }

        base = ours_sq;
        exp >>= 1usize;
        round += 1;
    }

    println!(
        "Our RTL exponent matches pairing: {}",
        acc == direct_pairing
    );

    if acc == direct_pairing {
        println!("ALL NATIVE DIAGNOSTICS PASSED.");
    }
}
