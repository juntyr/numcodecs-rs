//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.82.0-blue
//! [repo]: https://github.com/juntyr/numcodecs-rs
//!
//! [Latest Version]: https://img.shields.io/crates/v/numcodecs-wasm-host-reproducible
//! [crates.io]: https://crates.io/crates/numcodecs-wasm-host-reproducible
//!
//! [Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs-wasm-host-reproducible
//! [docs.rs]: https://docs.rs/numcodecs-wasm-host-reproducible/
//!
//! [Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/numcodecs-rs/numcodecs_wasm_host_reproducible
//!
//! Import a [`numcodecs`]-API-compatible [`DynCodec`][numcodecs::DynCodec]
//! from a WASM component inside a reproducible and fully sandboxed WebAssembly
//! runtime.

#![expect(clippy::multiple_crate_versions)] // FIXME

#[macro_use]
extern crate log;

mod codec;
mod engine;
mod logging;
mod stdio;
mod transform;

pub use codec::{ReproducibleWasmCodec, ReproducibleWasmCodecError, ReproducibleWasmCodecType};
