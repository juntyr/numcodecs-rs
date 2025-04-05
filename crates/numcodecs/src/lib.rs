//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.82.0-blue
//! [repo]: https://github.com/juntyr/numcodecs-rs
//!
//! [Latest Version]: https://img.shields.io/crates/v/numcodecs
//! [crates.io]: https://crates.io/crates/numcodecs
//!
//! [Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs
//! [docs.rs]: https://docs.rs/numcodecs/
//!
//! [Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/numcodecs-rs/numcodecs
//!
//! Compression codec API inspired by the [`numcodecs`] Python API.
//!
//! [`numcodecs`]: https://numcodecs.readthedocs.io/en/stable/

mod array;
mod codec;

pub use array::{
    AnyArcArray, AnyArray, AnyArrayAssignError, AnyArrayBase, AnyArrayDType, AnyArrayView,
    AnyArrayViewMut, AnyCowArray, AnyRawData, ArrayDType,
};
pub use codec::{
    codec_from_config_with_id, serialize_codec_config_with_id, Codec, DynCodec, DynCodecType,
    StaticCodec, StaticCodecConfig, StaticCodecType, StaticCodecVersion,
};

mod sealed {
    pub trait Sealed {}
}
