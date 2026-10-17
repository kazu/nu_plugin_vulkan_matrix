mod support;

use ndarray::{Array2, ArrayView2, array, s};
use support::gpu;
use vulkan_matrix_validation::{download, upload};

fn check(view: ArrayView2<'_, f64>) {
    let tensor = upload(gpu(), view).unwrap();
    assert_eq!(tensor.shape().as_slice(), view.shape());
    let actual = download::<f64>(gpu(), &tensor).unwrap();
    assert_eq!(actual.shape(), view.shape());
    for (a, b) in actual.iter().zip(view.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

#[test]
fn views_round_trip_without_changing_bits() {
    let values = Array2::from_shape_fn((6, 7), |(i, j)| (i * 7 + j) as f64);
    check(values.view());
    check(values.t());
    check(values.slice(s![1..5, ..]));
    check(values.slice(s![..;2, ..;3]));
    check(values.slice(s![..;-1, ..]));
    check(array![[f64::NAN, f64::INFINITY], [-0.0, f64::NEG_INFINITY]].view());
    let empty = Array2::<f64>::zeros((0, 3));
    check(empty.view());
    check(empty.t());
}

#[test]
fn uploaded_data_outlives_the_source() {
    let tensor = {
        let source = array![[1.0_f32, 2.0], [3.0, 4.0]];
        upload(gpu(), source.view()).unwrap()
    };
    assert_eq!(
        download::<f32>(gpu(), &tensor).unwrap(),
        array![[1.0, 2.0], [3.0, 4.0]]
    );
}
