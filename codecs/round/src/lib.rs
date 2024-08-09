//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.76.0-blue
//! [repo]: https://github.com/juntyr/numcodecs-rs
//!
//! [Latest Version]: https://img.shields.io/crates/v/numcodecs-round
//! [crates.io]: https://crates.io/crates/numcodecs-round
//!
//! [Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs-round
//! [docs.rs]: https://docs.rs/numcodecs-round/
//!
//! [Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/numcodecs-rs/numcodecs-round
//!
//! Rounding codec implementation for the [`numcodecs`] API.

use std::ops::{Div, Mul};

use ndarray::{Array, ArrayViewD, ArrayViewMutD, CowArray, Dimension};
use numcodecs::{
    AnyArray, AnyArrayDType, AnyArrayView, AnyArrayViewMut, AnyCowArray, Codec, StaticCodec,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

#[derive(Clone, Serialize, Deserialize)]
/// Codec that [`round`]s the data on encoding and passes through the input
/// unchanged during decoding.
///
/// The codec only supports floating point data.
pub struct RoundCodec {
    /// Precision of the rounding operation
    pub precision: Positive<f64>,
}

impl Codec for RoundCodec {
    type Error = RoundCodecError;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        match data {
            #[allow(clippy::cast_possible_truncation)]
            AnyCowArray::F32(data) => Ok(AnyArray::F32(round(
                data,
                Positive(self.precision.0 as f32),
            ))),
            AnyCowArray::F64(data) => Ok(AnyArray::F64(round(data, self.precision))),
            encoded => Err(RoundCodecError::UnsupportedDtype(encoded.dtype())),
        }
    }

    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
        match encoded {
            AnyCowArray::F32(encoded) => Ok(AnyArray::F32(encoded.into_owned())),
            AnyCowArray::F64(encoded) => Ok(AnyArray::F64(encoded.into_owned())),
            encoded => Err(RoundCodecError::UnsupportedDtype(encoded.dtype())),
        }
    }

    fn decode_into(
        &self,
        encoded: AnyArrayView,
        mut decoded: AnyArrayViewMut,
    ) -> Result<(), Self::Error> {
        fn shape_checked_assign<T: Copy>(
            encoded: &ArrayViewD<T>,
            decoded: &mut ArrayViewMutD<T>,
        ) -> Result<(), RoundCodecError> {
            #[allow(clippy::unit_arg)]
            if encoded.shape() == decoded.shape() {
                Ok(decoded.assign(encoded))
            } else {
                Err(RoundCodecError::MismatchedDecodeIntoShape {
                    decoded: encoded.shape().to_vec(),
                    provided: decoded.shape().to_vec(),
                })
            }
        }

        match (&encoded, &mut decoded) {
            (AnyArrayView::F32(encoded), AnyArrayViewMut::F32(decoded)) => {
                shape_checked_assign(encoded, decoded)
            }
            (AnyArrayView::F64(encoded), AnyArrayViewMut::F64(decoded)) => {
                shape_checked_assign(encoded, decoded)
            }
            (AnyArrayView::F32(_), decoded) => Err(RoundCodecError::MismatchedDecodeIntoDtype {
                decoded: AnyArrayDType::F32,
                provided: decoded.dtype(),
            }),
            (AnyArrayView::F64(_), decoded) => Err(RoundCodecError::MismatchedDecodeIntoDtype {
                decoded: AnyArrayDType::F64,
                provided: decoded.dtype(),
            }),
            (encoded, _decoded) => Err(RoundCodecError::UnsupportedDtype(encoded.dtype())),
        }
    }

    fn get_config<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.serialize(serializer)
    }
}

impl StaticCodec for RoundCodec {
    const CODEC_ID: &'static str = "round";

    fn from_config<'de, D: Deserializer<'de>>(config: D) -> Result<Self, D::Error> {
        Self::deserialize(config)
    }
}

#[allow(clippy::derive_partial_eq_without_eq)] // floats are not Eq
#[derive(Copy, Clone, PartialEq, PartialOrd, Hash)]
/// Positive floating point number
pub struct Positive<T: Float>(T);

impl serde::Serialize for Positive<f64> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_f64(self.0)
    }
}

impl<'de> serde::Deserialize<'de> for Positive<f64> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let x = f64::deserialize(deserializer)?;

        if x > 0.0 {
            Ok(Self(x))
        } else {
            Err(serde::de::Error::invalid_value(
                serde::de::Unexpected::Float(x),
                &"a positive value",
            ))
        }
    }
}

#[derive(Debug, Error)]
/// Errors that may occur when applying the [`RoundCodec`].
pub enum RoundCodecError {
    /// [`RoundCodec`] does not support the dtype
    #[error("Round does not support the dtype {0}")]
    UnsupportedDtype(AnyArrayDType),
    /// [`RoundCodec`] cannot decode the `decoded` dtype into the `provided`
    /// array
    #[error("Round cannot decode the dtype {decoded} into the provided {provided} array")]
    MismatchedDecodeIntoDtype {
        /// Dtype of the `decoded` data
        decoded: AnyArrayDType,
        /// Dtype of the `provided` array into which the data is to be decoded
        provided: AnyArrayDType,
    },
    /// [`RoundCodec`] cannot decode the decoded array into the provided
    /// array of a different shape
    #[error("Round cannot decode the decoded array of shape {decoded:?} into the provided array of shape {provided:?}")]
    MismatchedDecodeIntoShape {
        /// Shape of the `decoded` data
        decoded: Vec<usize>,
        /// Shape of the `provided` array into which the data is to be decoded
        provided: Vec<usize>,
    },
}

#[must_use]
/// Rounds the input `data` using `c = round(x / precision) * precision`
pub fn round<T: Float, D: Dimension>(data: CowArray<T, D>, precision: Positive<T>) -> Array<T, D> {
    let mut encoded = data.into_owned();
    encoded.mapv_inplace(|x| (x / precision.0).round() * precision.0);
    encoded
}

/// Floating point types
pub trait Float: Copy + Mul<Self, Output = Self> + Div<Self, Output = Self> {
    #[must_use]
    /// Returns the nearest integer to `self`. If a value is half-way between
    /// two integers, round away from 0.0.
    ///
    /// This method always returns the precise result.
    fn round(self) -> Self;
}

impl Float for f32 {
    fn round(self) -> Self {
        Self::round(self)
    }
}

impl Float for f64 {
    fn round(self) -> Self {
        Self::round(self)
    }
}
