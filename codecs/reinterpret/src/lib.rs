//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.76.0-blue
//! [repo]: https://github.com/juntyr/numcodecs-rs
//!
//! [Latest Version]: https://img.shields.io/crates/v/numcodecs-reinterpret
//! [crates.io]: https://crates.io/crates/numcodecs-reinterpret
//!
//! [Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs-reinterpret
//! [docs.rs]: https://docs.rs/numcodecs-reinterpret/
//!
//! [Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/numcodecs-rs/numcodecs-reinterpret
//!
//! Binary reinterpret codec implementation for the [`numcodecs`] API.

use numcodecs::{
    AnyArray, AnyArrayDType, AnyArrayView, AnyArrayViewMut, AnyCowArray, Codec, StaticCodec,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

#[derive(Clone)]
/// Codec to reinterpret data between different compatible types.
///
/// Note that no conversion happens, only the meaning of the bits changes.
pub struct ReinterpretCodec {
    encode_dtype: AnyArrayDType,
    decode_dtype: AnyArrayDType,
}

impl ReinterpretCodec {
    #[must_use]
    /// Try to create a [`ReinterpretCodec`] that reinterprets the input data
    /// from `decode_dtype` to `encode_dtype` on encoding, and from
    /// `encode_dtype` back to `decode_dtype` on decoding.
    ///
    /// Returns `Some(_)` if `encode_dtype` and `decode_dtype` are compatible,
    /// `None` otherwise.
    pub fn try_new(encode_dtype: AnyArrayDType, decode_dtype: AnyArrayDType) -> Option<Self> {
        #[allow(clippy::match_same_arms)]
        match (decode_dtype, encode_dtype) {
            // performing no conversion always works
            (ty_a, ty_b) if ty_a == ty_b => (),
            // converting to bytes always works
            (_, AnyArrayDType::U8) => (),
            // converting from signed / floating to same-size binary always works
            (AnyArrayDType::I16, AnyArrayDType::U16)
            | (AnyArrayDType::I32 | AnyArrayDType::F32, AnyArrayDType::U32)
            | (AnyArrayDType::I64 | AnyArrayDType::F64, AnyArrayDType::U64) => (),
            _ => return None,
        };

        Some(Self {
            encode_dtype,
            decode_dtype,
        })
    }

    #[must_use]
    /// Create a [`ReinterpretCodec`] that does not change the `dtype`.
    pub const fn passthrough(dtype: AnyArrayDType) -> Self {
        Self {
            encode_dtype: dtype,
            decode_dtype: dtype,
        }
    }

    #[must_use]
    /// Create a [`ReinterpretCodec`] that reinterprets `dtype` as
    /// [bytes][`AnyArrayDType::U8`].
    pub const fn to_bytes(dtype: AnyArrayDType) -> Self {
        Self {
            encode_dtype: AnyArrayDType::U8,
            decode_dtype: dtype,
        }
    }

    #[must_use]
    /// Create a  [`ReinterpretCodec`] that reinterprets `dtype` as its
    /// [binary][`AnyArrayDType::to_binary`] equivalent.
    pub const fn to_binary(dtype: AnyArrayDType) -> Self {
        Self {
            encode_dtype: dtype.to_binary(),
            decode_dtype: dtype,
        }
    }
}

impl Codec for ReinterpretCodec {
    type Error = ReinterpretCodecError;

    fn encode(&self, _data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        todo!()
    }

    fn decode(&self, _encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
        todo!()
    }

    fn decode_into(
        &self,
        _encoded: AnyArrayView,
        _decoded: AnyArrayViewMut,
    ) -> Result<(), Self::Error> {
        todo!()
    }

    fn get_config<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.serialize(serializer)
    }
}

impl StaticCodec for ReinterpretCodec {
    const CODEC_ID: &'static str = "reinterpret";

    fn from_config<'de, D: Deserializer<'de>>(config: D) -> Result<Self, D::Error> {
        Self::deserialize(config)
    }
}

impl Serialize for ReinterpretCodec {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        ReinterpretCodecConfig {
            encode_dtype: self.encode_dtype,
            decode_dtype: self.decode_dtype,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ReinterpretCodec {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let config = ReinterpretCodecConfig::deserialize(deserializer)?;

        #[allow(clippy::option_if_let_else)]
        match Self::try_new(config.encode_dtype, config.decode_dtype) {
            Some(codec) => Ok(codec),
            None => Err(serde::de::Error::custom(format!(
                "reinterpreting {} as {} is not allowed",
                config.decode_dtype, config.encode_dtype,
            ))),
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename = "ReinterpretCodec")]
struct ReinterpretCodecConfig {
    encode_dtype: AnyArrayDType,
    decode_dtype: AnyArrayDType,
}

#[derive(Debug, Error)]
/// Errors that may occur when applying the [`ReinterpretCodec`].
pub enum ReinterpretCodecError {
    /// [`ReinterpretCodec`] cannot decode the `decoded` dtype into the `provided`
    /// array
    #[error("Reinterpret cannot decode the dtype {decoded} into the provided {provided} array")]
    MismatchedDecodeIntoDtype {
        /// Dtype of the `decoded` data
        decoded: AnyArrayDType,
        /// Dtype of the `provided` array into which the data is to be decoded
        provided: AnyArrayDType,
    },
    /// [`ReinterpretCodec`] cannot decode the decoded array into the provided
    /// array of a different shape
    #[error("Reinterpret cannot decode the decoded array of shape {decoded:?} into the provided array of shape {provided:?}")]
    MismatchedDecodeIntoShape {
        /// Shape of the `decoded` data
        decoded: Vec<usize>,
        /// Shape of the `provided` array into which the data is to be decoded
        provided: Vec<usize>,
    },
}
