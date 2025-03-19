[![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![PyPi Release]][pypi] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs] [![Read the Docs]][rtdocs]

[CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
[workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain

[MSRV]: https://img.shields.io/badge/MSRV-1.82.0-blue
[repo]: https://github.com/juntyr/numcodecs-rs

[Latest Version]: https://img.shields.io/crates/v/numcodecs-zfp
[crates.io]: https://crates.io/crates/numcodecs-zfp

[PyPi Release]: https://img.shields.io/pypi/v/numcodecs-wasm-zfp.svg
[pypi]: https://pypi.python.org/pypi/numcodecs-wasm-zfp

[Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs-zfp
[docs.rs]: https://docs.rs/numcodecs-zfp/

[Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
[docs]: https://juntyr.github.io/numcodecs-rs/numcodecs_zfp

[Read the Docs]: https://img.shields.io/readthedocs/numcodecs-wasm?label=readthedocs
[rtdocs]: https://numcodecs-wasm.readthedocs.io/en/stable/api/numcodecs_wasm_zfp/

# numcodecs-zfp

ZFP codec implementation for the [`numcodecs`] API.

This implementation uses ZFP's [`ZFP_ROUNDING_MODE=ZFP_ROUND_FIRST`](https://zfp.readthedocs.io/en/release1.0.1/installation.html#c.ZFP_ROUNDING_MODE) and [`ZFP_WITH_TIGHT_ERROR=ON`](https://zfp.readthedocs.io/en/release1.0.1/installation.html#c.ZFP_WITH_TIGHT_ERROR) experimental features to reduce the bias and correlation in ZFP's errors (see <https://zfp.readthedocs.io/en/release1.0.1/faq.html#zfp-rounding>).

This implementation also rejects non-reversibly compressing non-finite (infinite or NaN) values, since ZFP's behaviour for them is undefined (see <https://zfp.readthedocs.io/en/release1.0.1/faq.html#q-valid>).

Please see the `numcodecs-zfp-classic` codec for an implementation that uses ZFP without these modifications.

[`numcodecs`]: https://docs.rs/numcodecs/0.2/numcodecs/

## License

Licensed under the Mozilla Public License, Version 2.0 ([LICENSE](LICENSE) or https://www.mozilla.org/en-US/MPL/2.0/).

## Funding

The `numcodecs-zfp` crate has been developed as part of [ESiWACE3](https://www.esiwace.eu), the third phase of the Centre of Excellence in Simulation of Weather and Climate in Europe.

Funded by the European Union. This work has received funding from the European High Performance Computing Joint Undertaking (JU) under grant agreement No 101093054.
