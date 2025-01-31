[![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]

[CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
[workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain

[MSRV]: https://img.shields.io/badge/MSRV-1.82.0-blue
[repo]: https://github.com/juntyr/numcodecs-rs

[Latest Version]: https://img.shields.io/crates/v/numcodecs-wasm-host-reproducible
[crates.io]: https://crates.io/crates/numcodecs-wasm-host-reproducible

[Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs-wasm-host-reproducible
[docs.rs]: https://docs.rs/numcodecs-wasm-host-reproducible/

[Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
[docs]: https://juntyr.github.io/numcodecs-rs/numcodecs_wasm_host_reproducible

# numcodecs-wasm-host-reproducible

Import a [`numcodecs`]-API-compatible [`DynCodec`] from a WASM component inside a reproducible and fully sandboxed WebAssembly runtime.

[`numcodecs`]: https://docs.rs/numcodecs/0.2/numcodecs/
[`DynCodec`]: https://docs.rs/numcodecs/latest/numcodecs/trait.DynCodec.html

## License

Licensed under the Mozilla Public License, Version 2.0 ([LICENSE](LICENSE) or https://www.mozilla.org/en-US/MPL/2.0/).

## Funding

The `numcodecs-wasm-host-reproducible` crate has been developed as part of [ESiWACE3](https://www.esiwace.eu), the third phase of the Centre of Excellence in Simulation of Weather and Climate in Europe.

Funded by the European Union. This work has received funding from the European High Performance Computing Joint Undertaking (JU) under grant agreement No 101093054.
