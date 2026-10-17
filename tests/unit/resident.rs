use crate::gpu::{download_array, upload_array};
use crate::resident::Resources;
use ndarray::{ArrayD, IxDyn};
use nu_protocol::{Span, Value};
use std::sync::{Arc, Barrier};

#[test]
fn handles_keep_metadata_and_inflight_resources_without_reusing_ids() {
    let span = Span::unknown();
    let gpu = crate::test_gpu::gpu().clone();
    let mut resources = Resources::new().unwrap();
    let mut restarted = Resources::new().unwrap();
    let make = || upload_array(&gpu, ArrayD::from_elem(IxDyn(&[2, 3]), 7.0_f64)).unwrap();
    let value = resources
        .insert(gpu.clone(), make(), "d0", "f64", span)
        .unwrap();
    let cloned = value.clone();
    let metadata = value
        .as_custom_value()
        .unwrap()
        .to_base_value(span)
        .unwrap();
    assert_eq!(
        metadata.as_record().unwrap().get("dtype").unwrap(),
        &Value::string("f64", span)
    );
    let held = resources.get(&cloned, span).unwrap();
    let weak = Arc::downgrade(&held);
    let barrier = Arc::new(Barrier::new(2));
    let worker_barrier = barrier.clone();
    let worker = std::thread::spawn(move || {
        worker_barrier.wait();
        download_array::<f64>(&held.gpu, &held.tensor).unwrap()
    });
    resources.remove(value.as_custom_value().unwrap());
    assert!(resources.is_empty());
    assert!(resources.get(&cloned, span).is_err());
    barrier.wait();
    assert_eq!(
        worker.join().unwrap(),
        ArrayD::from_elem(IxDyn(&[2, 3]), 7.0)
    );
    assert!(weak.upgrade().is_none());
    let next = resources
        .insert(gpu.clone(), make(), "d0", "f64", span)
        .unwrap();
    assert!(resources.get(&value, span).is_err());
    assert!(resources.get(&next, span).is_ok());
    let fresh = restarted
        .insert(gpu.clone(), make(), "d0", "f64", span)
        .unwrap();
    assert!(restarted.get(&value, span).is_err());
    assert!(resources.get(&fresh, span).is_err());
    restarted.remove(value.as_custom_value().unwrap());
    assert!(restarted.get(&fresh, span).is_ok());
    resources.remove(fresh.as_custom_value().unwrap());
    assert!(!resources.is_empty());
    assert!(next.as_custom_value().unwrap().notify_plugin_on_drop());
}
