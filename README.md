# nu_plugin_vulkan_matrix

A [Nushell][nushell] plugin for GPU matrix operations using Vulkan, with an interface similar to Nushell's [matrix][matrix].

## Features

- Run matrix multiplication, elementwise operations, reshaping, and reductions on the GPU with operations similar to Nu's `matrix`.
- Discrete GPUs generally start to show gains with matrices of about 512×512 or larger. Integrated GPUs can also be faster with smaller matrices.
- Keep data on the GPU across a chain of operations. CPU-only processing is faster when transfers between memory and VRAM cost more than the time saved on computation.
- Read from the Linux kernel's page cache into VRAM and write from VRAM back to the page cache, using NumPy's NPY file format.
- In shared-memory environments such as integrated GPUs, `--shared` skips the input copy. On AMD Radeon 780M (integrated in Ryzen 9 8945HS), NPY loading plus matrix multiplication dropped from **72.1 to 44.0 ms**.

Example f64 matrix multiplication timings (including NPY file loading, mean of 10 runs, in ms):

| Matrix size | Scope | CPU (Nu matrix) | RX 7800 XT | Integrated Radeon 780M | Radeon 780M shared input |
|---|---|---:|---:|---:|---:|
| 256×256 | File loading + matrix multiplication | 18.1 | 39.8 | 8.7 | 5.3 |
| 512×512 | File loading + matrix multiplication | 66.7 | 40.6 | 17.6 | 14.1 |
| 1024×1024 | File loading + matrix multiplication | 255.7 | 53.0 | 72.1 | 44.0 |
| 256×256 | Above + output to Nu | 20.4 | 60.8 | 27.3 | 23.7 |
| 512×512 | Above + output to Nu | 76.8 | 102.4 | 90.6 | 89.2 |
| 1024×1024 | Above + output to Nu | 301.5 | 348.5 | 363.3 | 352.5 |

See [benchmarks][benchmarks] for measurement conditions and results for all GPUs, and [NPY I/O][file-io] for shared-input requirements.

## Installation

Requires Nu 0.116.1, Rust 1.96.1 (`rust-toolchain.toml`), and a Vulkan GPU that preserves subnormal values
for the selected dtype. The backend is bundled with the plugin.

Install from crates.io (after the first publication, run in Nushell):

```nu
cargo install nu_plugin_vulkan_matrix --locked --root $env.HOME
plugin add ~/bin/nu_plugin_vulkan_matrix
plugin use vulkan_matrix
```

To install from the GitHub source, replace the first command with:

```nu
cargo install --git https://github.com/kazu/nu_plugin_vulkan_matrix --locked --root $env.HOME
```

## Usage

List the available GPUs.

Nushell:

```nu
vulkan matrix devices
```

Replace `d0` in the examples below with a GPU from the device list.

Chain matrix multiplication and elementwise operations on the GPU, then retrieve the result. The default dtype is f64.

Nushell:

```nu
let a = [[1 2] [3 4]] | vulkan matrix from-nu --device d0
let b = [[5 6] [7 8]] | vulkan matrix from-nu --device d0
let result = $a | vulkan matrix multiply $b | vulkan matrix scale 2
$result | vulkan matrix into-nu
```

`$result` is a matrix on the GPU. `into-nu` retrieves all its elements.
`+ - * /` are elementwise operations; `multiply` performs matrix multiplication.
See [operations][operations] for conversions to and from Nu's `matrix`, reductions, and other operations.

On Linux, save results to NumPy NPY files and load them back onto the GPU.

Nushell (continuing the example above):

```nu
$result | vulkan matrix to-npy ./result.npy
let restored = vulkan matrix from-npy ./result.npy --device d0
$restored | vulkan matrix into-nu
```

Writing to an existing file returns an error.
See [NPY I/O][file-io] for supported formats, shared input, and driver requirements.

## Development

Run all checks with `make ci` (Nushell / POSIX shell; requires GNU Make).
See [development and validation][development] for prerequisites, source layout, and validation and benchmark procedures.

## Layout

- `src/`: Plugin and GPU operations.
- `tests/`: Rust / Nushell tests.
- `validation/`: Hardware validation and benchmarks.
- `backend/`: Project-specific Vulkan backend. See [origin and changes][backend].
- `docs/`: Details on operations, file I/O, and development.

## Related

- [Nushell][nushell]: A shell for structured data.
- [matrix][matrix]: Nushell's built-in matrix operations.
- [polars-dyn][polars-dyn]: Planned improvements to integration between DataFrames and GPU matrix operations.

## License

[MIT][license] © 2026 Kazuhisa TAKEI. The CubeCL-derived backend retains its original copyright notices and licenses.

[nushell]: https://www.nushell.sh/
[matrix]: https://www.nushell.sh/commands/docs/matrix.html
[polars-dyn]: https://github.com/kazu/nu_plugin_polars_dyn
[benchmarks]: docs/benchmarks.md
[file-io]: docs/file-io.md
[operations]: docs/operations.md
[development]: docs/development.md
[backend]: backend/README.md
[license]: LICENSE
