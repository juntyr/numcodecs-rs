//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.64.0-blue
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

use std::{error::Error, marker::PhantomData};

use serde::{Deserializer, Serializer};

mod array;

pub use array::{
    AnyArcArray, AnyArray, AnyArrayBase, AnyArrayDType, AnyArrayView, AnyArrayViewMut, AnyCowArray,
    AnyRawData,
};

/// Compression codec that [`encode`][`Codec::encode`]s and
/// [`decode`][`Codec::decode`]s numeric n-dimensional arrays.
pub trait Codec: 'static + Send + Sync + Clone {
    /// Error type that may be returned during [`encode`][`Codec::encode`]ing
    /// and [`decode`][`Codec::decode`]ing.
    type Error: 'static + Send + Sync + Error;

    /// Encodes the `data` and returns the result.
    ///
    /// # Errors
    ///
    /// Errors if encoding the buffer fails.
    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error>;

    /// Decodes the `encoded` data and returns the result.
    ///
    /// # Errors
    ///
    /// Errors if decoding the buffer fails.
    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, Self::Error>;

    /// Decodes the `encoded` data and writes the result into the provided
    /// `decoded` output.
    ///
    /// The output must have the correct type and shape.
    ///
    /// # Errors
    ///
    /// Errors if decoding the buffer fails.
    fn decode_into(
        &self,
        encoded: AnyArrayView,
        decoded: AnyArrayViewMut,
    ) -> Result<(), Self::Error>;

    /// Serializes the configuration parameters for this codec.
    ///
    /// The config must include an `id` field with the [`DynCodecType::codec_id`].
    /// The config must be compatible with JSON encoding.
    ///
    /// # Errors
    ///
    /// Errors if serializing the codec configuration fails.
    fn get_config<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error>;
}

/// Statically typed compression codec.
pub trait StaticCodec: Codec {
    /// Codec identifier.
    const CODEC_ID: &'static str;

    /// Instantiate a codec from a serialized `config`uration.
    ///
    /// The config must be compatible with JSON encoding.
    ///
    /// # Errors
    ///
    /// Errors if constructing the codec fails.
    fn from_config<'de, D: Deserializer<'de>>(config: D) -> Result<Self, D::Error>;
}

/// Dynamically typed compression codec.
///
/// Every codec that implements [`StaticCodec`] also implements [`DynCodec`].
pub trait DynCodec: Codec {
    /// Type object type for this codec.
    type Type: DynCodecType;

    /// Returns the type object for this codec.
    fn ty(&self) -> Self::Type;
}

/// Type object for dynamically typed compression codecs.
pub trait DynCodecType: 'static + Send + Sync {
    /// Type of the instances of this codec type object.
    type Codec: DynCodec<Type = Self>;

    /// Codec identifier.
    fn codec_id(&self) -> &str;

    /// Instantiate a codec of this type from a serialized `config`uration.
    ///
    /// The config must be compatible with JSON encoding.
    ///
    /// # Errors
    ///
    /// Errors if constructing the codec fails.
    fn codec_from_config<'de, D: Deserializer<'de>>(
        &self,
        config: D,
    ) -> Result<Self::Codec, D::Error>;
}

impl<T: StaticCodec> DynCodec for T {
    type Type = StaticCodecType<Self>;

    fn ty(&self) -> Self::Type {
        StaticCodecType::of()
    }
}

/// Type object for statically typed compression codecs.
pub struct StaticCodecType<T: StaticCodec> {
    _marker: PhantomData<T>,
}

impl<T: StaticCodec> StaticCodecType<T> {
    /// Statically obtain the type for a statically typed codec.
    #[must_use]
    pub const fn of() -> Self {
        Self {
            _marker: PhantomData::<T>,
        }
    }
}

impl<T: StaticCodec> DynCodecType for StaticCodecType<T> {
    type Codec = T;

    fn codec_id(&self) -> &str {
        T::CODEC_ID
    }

    fn codec_from_config<'de, D: Deserializer<'de>>(
        &self,
        config: D,
    ) -> Result<Self::Codec, D::Error> {
        T::from_config(config)
    }
}
