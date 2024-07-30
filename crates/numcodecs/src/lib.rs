//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.65.0-blue
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

use std::{borrow::Cow, error::Error};

use ndarray::{CowArray, IxDyn};
use serde::{Deserializer, Serializer};

/// Compression codec that [`encode`][`Codec::encode`]s and
/// [`decode`][`Codec::decode`]s numeric n-dimensional arrays.
pub trait Codec: Clone {
    /// Error type that may be returned during [`encode`][`Codec::encode`]ing
    /// and [`decode`][`Codec::decode`]ing.
    type Error: 'static + Send + Sync + Error;

    /// Instantiate a codec from a serialized `config`uration.
    ///
    /// The config must be compatible with JSON encoding.
    ///
    /// # Errors
    ///
    /// Errors if constructing the codec fails.
    fn from_config<'de, D: Deserializer<'de>>(config: D) -> Result<Self, D::Error>;

    /// Encodes the `data` and returns the result.
    ///
    /// # Errors
    ///
    /// Errors if encoding the buffer fails.
    fn encode<'a>(&self, data: AnyCowArray<'a>) -> Result<AnyCowArray<'a>, Self::Error>;

    /// Decodes the `encoded` data and returns the result.
    ///
    /// # Errors
    ///
    /// Errors if encoding the buffer fails.
    fn decode<'a>(&self, encoded: AnyCowArray<'a>) -> Result<AnyCowArray<'a>, Self::Error>;

    /// Serializes the configuration parameters for this codec.
    ///
    /// The config must include an `id` field with the [`DynCodec::codec_id`].
    /// The config must be compatible with JSON encoding.
    ///
    /// # Errors
    ///
    /// Errors if serializing the codec configuration fails.
    fn get_config<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error>;
}

/// Numeric n-dimensional arrays with dynamic shapes.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum AnyCowArray<'a> {
    U8(CowArray<'a, u8, IxDyn>),
    U16(CowArray<'a, u16, IxDyn>),
    U32(CowArray<'a, u32, IxDyn>),
    U64(CowArray<'a, u64, IxDyn>),
    I8(CowArray<'a, i8, IxDyn>),
    I16(CowArray<'a, i16, IxDyn>),
    I32(CowArray<'a, i32, IxDyn>),
    I64(CowArray<'a, i64, IxDyn>),
    F32(CowArray<'a, f32, IxDyn>),
    F64(CowArray<'a, f64, IxDyn>),
}

/// Compression codec whose [`CODEC_ID`](`StaticCodec::CODEC_ID`) is statically
/// known.
pub trait StaticCodec: 'static + Codec {
    /// Codec identifier.
    const CODEC_ID: &'static str;
}

/// Compression codec whose [`codec_id`](`DynCodec::codec_id`) is dynamically
/// known.
///
/// Every codec that implements [`StaticCodec`] also implements [`DynCodec`].
pub trait DynCodec: Codec {
    /// Codec identifier.
    fn codec_id(&self) -> Cow<str>;
}

impl<T: StaticCodec> DynCodec for T {
    fn codec_id(&self) -> Cow<str> {
        Cow::Borrowed(T::CODEC_ID)
    }
}
