use crate::{
    elementwise::{self, Arithmetic, Operand},
    gpu::{Gpu, download_array, upload_view},
    layout,
};
use ndarray::{ArrayD, IxDyn};

fn gpu() -> &'static Gpu {
    crate::test_gpu::gpu()
}

#[test]
fn generation_and_strided_arithmetic_match_ndarray() {
    let gpu = gpu();
    for shape in [&[3][..], &[2, 3, 4], &[2, 0, 3]] {
        let zeros = elementwise::create::<f64>(gpu, shape, false).unwrap();
        assert_eq!(
            download_array::<f64>(gpu, &zeros).unwrap(),
            ArrayD::<f64>::zeros(IxDyn(shape))
        );
    }
    for n in [1, 3, 4] {
        let identity = elementwise::create::<f32>(gpu, &[n, n], true).unwrap();
        assert_eq!(
            download_array::<f32>(gpu, &identity).unwrap(),
            ndarray::Array2::<f32>::eye(n).into_dyn()
        );
    }
    let a = ArrayD::from_shape_vec(IxDyn(&[2, 3, 4]), (0..24).map(|n| n as f64 - 12.).collect())
        .unwrap();
    let input = upload_view(gpu, a.view()).unwrap();
    let transposed = layout::transpose(&input);
    for (tensor, array) in [(&input, a.view()), (&transposed, a.view().reversed_axes())] {
        let b = array.mapv(|x| x * 0.25 + 8.);
        let rhs = upload_view(gpu, b.view()).unwrap();
        for op in [
            Arithmetic::Add,
            Arithmetic::Subtract,
            Arithmetic::Multiply,
            Arithmetic::Divide,
        ] {
            let result = elementwise::apply(gpu, tensor, Operand::Tensor(&rhs), op, false).unwrap();
            let expected = match op {
                Arithmetic::Add => &array + &b,
                Arithmetic::Subtract => &array - &b,
                Arithmetic::Multiply => &array * &b,
                Arithmetic::Divide => &array / &b,
            };
            let actual = download_array::<f64>(gpu, &result).unwrap();
            assert_eq!(actual.shape(), expected.shape());
            for (&x, &y) in actual.iter().zip(expected.iter()) {
                if matches!(op, Arithmetic::Divide) {
                    assert!((x - y).abs() <= 4. * f64::EPSILON * y.abs().max(1.));
                } else {
                    assert_eq!(x, y);
                }
            }
        }
        let scaled = elementwise::apply(
            gpu,
            tensor,
            Operand::Scalar(2.),
            Arithmetic::Multiply,
            false,
        )
        .unwrap();
        assert_eq!(download_array::<f64>(gpu, &scaled).unwrap(), &array * 2.);
    }
    let row = ndarray::array![1., 2., 3., 4.].into_dyn();
    let rhs = upload_view(gpu, row.view()).unwrap();
    assert!(
        elementwise::apply(gpu, &input, Operand::Tensor(&rhs), Arithmetic::Add, false).is_err()
    );
    let result =
        elementwise::apply(gpu, &input, Operand::Tensor(&rhs), Arithmetic::Add, true).unwrap();
    assert_eq!(
        download_array::<f64>(gpu, &result).unwrap(),
        &a + &row.broadcast(a.raw_dim()).unwrap()
    );
    assert!(elementwise::apply(gpu, &rhs, Operand::Tensor(&input), Arithmetic::Add, true).is_err());
    assert_eq!(download_array::<f64>(gpu, &input).unwrap(), a);
    let rhs32 = upload_view(gpu, row.mapv(|x| x as f32).view()).unwrap();
    assert!(
        elementwise::apply(gpu, &rhs, Operand::Tensor(&rhs32), Arithmetic::Add, false).is_err()
    );
    for shape in [&[][..], &[2, 0, 3]] {
        let a = ArrayD::<f64>::from_elem(IxDyn(shape), -0.0);
        let tensor = upload_view(gpu, a.view()).unwrap();
        let result = elementwise::apply(
            gpu,
            &tensor,
            Operand::Scalar(2.),
            Arithmetic::Multiply,
            false,
        )
        .unwrap();
        assert_eq!(download_array::<f64>(gpu, &result).unwrap(), &a * 2.);
    }
    let incompatible =
        upload_view(gpu, ndarray::Array2::<f64>::zeros((2, 5)).into_dyn().view()).unwrap();
    assert!(
        elementwise::apply(
            gpu,
            &input,
            Operand::Tensor(&incompatible),
            Arithmetic::Add,
            true
        )
        .is_err()
    );
}

#[test]
fn ieee_values_are_not_narrowed_or_flushed() {
    macro_rules! check {
        ($t:ty) => {{
            let a = ndarray::array![
                0.0 as $t,
                -0.0,
                <$t>::INFINITY,
                <$t>::NEG_INFINITY,
                <$t>::NAN,
                <$t>::from_bits(1),
                1.0000000000000002 as $t
            ]
            .into_dyn();
            let tensor = upload_view(gpu(), a.view()).unwrap();
            for (op, scalar) in [
                (Arithmetic::Multiply, 1.),
                (Arithmetic::Divide, -1.),
                (Arithmetic::Add, 0.),
                (Arithmetic::Subtract, 0.),
            ] {
                let output =
                    elementwise::apply(gpu(), &tensor, Operand::Scalar(scalar), op, false).unwrap();
                let actual = download_array::<$t>(gpu(), &output).unwrap();
                for (&x, &y) in a.iter().zip(actual.iter()) {
                    let expected = match op {
                        Arithmetic::Add => x + scalar as $t,
                        Arithmetic::Subtract => x - scalar as $t,
                        Arithmetic::Multiply => x * scalar as $t,
                        Arithmetic::Divide => x / scalar as $t,
                    };
                    assert!(
                        if expected.is_nan() {
                            y.is_nan()
                        } else {
                            y.to_bits() == expected.to_bits()
                        },
                        "{op:?}: {x:?} -> {y:?}, expected {expected:?}"
                    );
                }
            }
        }};
    }
    check!(f64);
    check!(f32);
}
