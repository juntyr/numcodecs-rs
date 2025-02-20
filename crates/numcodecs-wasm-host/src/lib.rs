//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.82.0-blue
//! [repo]: https://github.com/juntyr/numcodecs-rs
//!
//! [Latest Version]: https://img.shields.io/crates/v/numcodecs-wasm-host
//! [crates.io]: https://crates.io/crates/numcodecs-wasm-host
//!
//! [Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs-wasm-host
//! [docs.rs]: https://docs.rs/numcodecs-wasm-host/
//!
//! [Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/numcodecs-rs/numcodecs_wasm_host
//!
//! wasm32 host-side bindings for the [`numcodecs`] API, which allows you to
//! import a codec from a WASM component.

// postcard depends on embedded-io 0.4 and 0.6
#![allow(clippy::multiple_crate_versions)]

mod codec;
mod component;
mod error;
mod wit;

pub use codec::WasmCodec;
pub use component::WasmCodecComponent;
pub use error::{CodecError, RuntimeError};
pub use wit::NumcodecsWitInterfaces;
