[![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![PyPi Release]][pypi] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs] [![Read the Docs]][rtdocs]

[CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
[workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain

[MSRV]: https://img.shields.io/badge/MSRV-1.85.0-blue
[repo]: https://github.com/juntyr/numcodecs-rs

[Latest Version]: https://img.shields.io/crates/v/numcodecs-zfp-classic
[crates.io]: https://crates.io/crates/numcodecs-zfp-classic

[PyPi Release]: https://img.shields.io/pypi/v/numcodecs-wasm-zfp-classic.svg
[pypi]: https://pypi.python.org/pypi/numcodecs-wasm-zfp-classic

[Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs-zfp-classic
[docs.rs]: https://docs.rs/numcodecs-zfp-classic/

[Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
[docs]: https://juntyr.github.io/numcodecs-rs/numcodecs_zfp_classic

[Read the Docs]: https://img.shields.io/readthedocs/numcodecs-wasm?label=readthedocs
[rtdocs]: https://numcodecs-wasm.readthedocs.io/en/stable/api/numcodecs_wasm_zfp_classic/

# numcodecs-zfp-classic

ZFP (classic) codec implementation for the [`numcodecs`] API.

This implementation uses ZFP's default [`ZFP_ROUNDING_MODE=ZFP_ROUND_NEVER`](https://zfp.readthedocs.io/en/release1.0.1/installation.html#c.ZFP_ROUNDING_MODE) rounding mode, which is known to increase bias and correlation in ZFP's errors (see <https://zfp.readthedocs.io/en/release1.0.1/faq.html#zfp-rounding>).

Please see the `numcodecs-zfp` codec for an implementation that uses an improved version of ZFP.

[`numcodecs`]: https://docs.rs/numcodecs/0.2/numcodecs/

## License

Licensed under the Mozilla Public License, Version 2.0 ([LICENSE](LICENSE) or https://www.mozilla.org/en-US/MPL/2.0/).

## Funding

The `numcodecs-zfp-classic` crate has been developed as part of [ESiWACE3](https://www.esiwace.eu), the third phase of the Centre of Excellence in Simulation of Weather and Climate in Europe.

Funded by the European Union. This work has received funding from the European High Performance Computing Joint Undertaking (JU) under grant agreement No 101093054.
