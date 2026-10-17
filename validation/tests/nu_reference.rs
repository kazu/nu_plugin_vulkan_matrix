#[test]
fn native_matrix_preserves_the_f64_sentinel() {
    let output = std::process::Command::new("nu")
        .args([
            "-c",
            r#"
            [[16777217.0 1.0 -16777216.0] [2.0 3.0 4.0]]
            | into matrix
            | matrix multiply ([[1.0 0.0] [1.0 1.0] [1.0 0.0]] | into matrix)
            | matrix into-nu
            | to nuon
        "#,
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout).unwrap().trim(),
        "[[2.0, 1.0], [9.0, 3.0]]"
    );
}
