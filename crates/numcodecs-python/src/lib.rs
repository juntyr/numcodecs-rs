//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.82.0-blue
//! [repo]: https://github.com/juntyr/numcodecs-rs
//!
//! [Latest Version]: https://img.shields.io/crates/v/numcodecs-python
//! [crates.io]: https://crates.io/crates/numcodecs-python
//!
//! [Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs-python
//! [docs.rs]: https://docs.rs/numcodecs-python/
//!
//! [Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/numcodecs-rs/numcodecs_python
//!
//! Rust-bindings for the [`numcodecs`] Python API using [`pyo3`].
//!
//! [`numcodecs`]: https://numcodecs.readthedocs.io/en/stable/
//! [`pyo3`]: https://docs.rs/pyo3/0.23/pyo3/

#[cfg(test)]
use ::serde_json as _;

mod adapter;
mod codec;
mod codec_class;
mod export;
mod registry;
mod schema;
mod utils;

pub use adapter::{PyCodecAdapter, PyCodecClassAdapter};
pub use codec::{PyCodec, PyCodecMethods};
pub use codec_class::{PyCodecClass, PyCodecClassMethods};
pub use export::export_codec_class;
pub use registry::PyCodecRegistry;

mod sealed {
    pub trait Sealed {}
}
