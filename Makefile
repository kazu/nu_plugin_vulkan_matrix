SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
export CARGO_TARGET_DIR ?= $(HOME)/.cargo/target
export RAYON_NUM_THREADS := 1
export MATMUL_NUM_THREADS := 1

.PHONY: ci ci-host backend-check package package-backend publish publish-backend __git_task_ci_probe verify bench bench-elements bench-reductions bench-gpu
NU ?= nu
NU_ARGS ?=
package-backend:
	cargo package --manifest-path backend/Cargo.toml --locked

package:
	cargo package --locked

publish-backend:
	cargo publish --manifest-path backend/Cargo.toml --locked

publish:
	cargo publish --locked

backend-check:
	cargo fmt --manifest-path backend/Cargo.toml --check
	cargo clippy --manifest-path backend/Cargo.toml --locked --all-targets --no-default-features --features std,spirv -- -D warnings
	RUSTDOCFLAGS='-D warnings' cargo doc --manifest-path backend/Cargo.toml --locked --no-deps --no-default-features --features std,spirv

# Hosted runners have no Vulkan device. Execute the GPU tests with `make ci` locally.
ci-host: backend-check
	cargo fmt --check
	cargo clippy --locked --all-targets -- -D warnings
	cargo test --locked --no-run
	RUSTDOCFLAGS='-D warnings' cargo doc --locked --no-deps
	cargo fmt --manifest-path validation/Cargo.toml --check
	cargo clippy --manifest-path validation/Cargo.toml --locked --all-targets -- -D warnings
	cargo test --manifest-path validation/Cargo.toml --locked --no-run
	RUSTDOCFLAGS='-D warnings' cargo doc --manifest-path validation/Cargo.toml --locked --no-deps
	cargo bench --manifest-path validation/Cargo.toml --locked --no-run

ci:
	task_ci_max=0
	task_ci_tmp=$$(mktemp)
	trap 'rm -f "$$task_ci_tmp" "$$task_ci_tmp.block"' EXIT
	ci_step() {
		local name="$$1" code=0
		shift
		"$$@" > "$$task_ci_tmp" 2>&1 || code=$$?
		if (( code > task_ci_max )); then task_ci_max=$$code; fi
		{ printf '===== %s (exit %s) =====\n' "$$name" "$$code"; cat "$$task_ci_tmp"; printf '\n'; } > "$$task_ci_tmp.block"
		if [[ -n "$${CI_LOG:-}" ]]; then cat "$$task_ci_tmp.block" >> "$$CI_LOG"; else cat "$$task_ci_tmp.block"; fi
	}
	ci_step root-fmt cargo fmt --check
	ci_step backend-check $(MAKE) backend-check
	ci_step root-clippy cargo clippy --locked --all-targets -- -D warnings
	ci_step root-test cargo test --locked --no-fail-fast -- --nocapture
	ci_step root-doc env RUSTDOCFLAGS='-D warnings' cargo doc --locked --no-deps
	cd validation
	ci_step validation-fmt cargo fmt --check
	ci_step validation-clippy cargo clippy --locked --all-targets -- -D warnings
	ci_step validation-test cargo test --locked --no-fail-fast -- --nocapture
	ci_step validation-doc env RUSTDOCFLAGS='-D warnings' cargo doc --locked --no-deps
	ci_step validation-bench-build cargo bench --locked --no-run
	if [[ -n "$${CI_MAX_FILE:-}" ]]; then printf '%s\n' "$$task_ci_max" > "$$CI_MAX_FILE"; fi
	exit $$task_ci_max

__git_task_ci_probe: ; @true

verify:
	cd validation
	cargo test --locked --no-fail-fast -- --nocapture

bench:
	$(NU) $(NU_ARGS) validation/benches/nu-matrix.nu

bench-gpu:
	cd validation
	cargo bench --locked --bench baseline

bench-elements:
	$(NU) $(NU_ARGS) validation/benches/nu-elementwise.nu

bench-reductions:
	$(NU) $(NU_ARGS) validation/benches/nu-reductions.nu
