[![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]

[CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
[workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain

[MSRV]: https://img.shields.io/badge/MSRV-1.85.0-blue
[repo]: https://github.com/juntyr/numcodecs-rs

[Latest Version]: https://img.shields.io/crates/v/numcodecs-wasm-guest
[crates.io]: https://crates.io/crates/numcodecs-wasm-guest

[Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs-wasm-guest
[docs.rs]: https://docs.rs/numcodecs-wasm-guest/

[Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
[docs]: https://juntyr.github.io/numcodecs-rs/numcodecs_wasm_guest

# numcodecs-wasm-guest

wasm32 guest-side bindings for the [`numcodecs`] API, which allows you to export one [`StaticCodec`] from a WASM component.

[`numcodecs`]: https://docs.rs/numcodecs/0.2/numcodecs/
[`StaticCodec`]: https://docs.rs/numcodecs/latest/numcodecs/trait.StaticCodec.html

## License

Licensed under the Mozilla Public License, Version 2.0 ([LICENSE](LICENSE) or https://www.mozilla.org/en-US/MPL/2.0/).

## Funding

The `numcodecs-wasm-guest` crate has been developed as part of [ESiWACE3](https://www.esiwace.eu), the third phase of the Centre of Excellence in Simulation of Weather and Climate in Europe.

Funded by the European Union. This work has received funding from the European High Performance Computing Joint Undertaking (JU) under grant agreement No 101093054.
