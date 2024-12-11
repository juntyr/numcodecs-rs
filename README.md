[![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Main]][docs]

[CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
[workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain

[MSRV]: https://img.shields.io/badge/MSRV-1.81.0-blue
[repo]: https://github.com/juntyr/numcodecs-rs

[Latest Version]: https://img.shields.io/crates/v/numcodecs
[crates.io]: https://crates.io/search?q=numcodecs

[Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
[docs]: https://juntyr.github.io/numcodecs-rs
# numcodecs-rs

This repository provides a compression codec API in Rust inspired by the [`numcodecs`] Python API.

The repository is structured as follows:

- [`crates`](crates): Fundamental compression API crates
  - [`numcodecs`](crates/numcodecs): Rusty compression codec API
  - [`numcodecs-python`](crates/numcodecs-python): Rust bindings to the [`numcodecs`] Python API, which allows Python codecs to be used in Rust and Rust codecs to be used in Python
  - [`numcodecs-wasm-guest`](crates/numcodecs-wasm-guest): Export a Rust codec as a WebAssembly [component] when compiling to WebAssembly
- [`codecs`](codecs): Codec implementation crates, some new, some adapting existing [`numcodecs`] codecs with a more composable API

[`numcodecs`]: https://numcodecs.readthedocs.io/en/stable/
[component]: https://component-model.bytecodealliance.org/design/components.html

## License

Licensed under the Mozilla Public License, Version 2.0 ([LICENSE](LICENSE) or https://www.mozilla.org/en-US/MPL/2.0/).

## Funding

The `numcodecs-rs` repository has been developed as part of [ESiWACE3](https://www.esiwace.eu), the third phase of the Centre of Excellence in Simulation of Weather and Climate in Europe.

Funded by the European Union. This work has received funding from the European High Performance Computing Joint Undertaking (JU) under grant agreement No 101093054.
