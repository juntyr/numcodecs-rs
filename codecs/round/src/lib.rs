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
//! [docs]: https://juntyr.github.io/numcodecs-rs/numcodecs_round
//!
//! Rounding codec implementation for the [`numcodecs`] API.

use std::ops::{Div, Mul};

use ndarray::{Array, ArrayBase, Data, Dimension};
use numcodecs::{
    AnyArray, AnyArrayAssignError, AnyArrayDType, AnyArrayView, AnyArrayViewMut, AnyCowArray,
    Codec, StaticCodec,
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
        if !matches!(encoded.dtype(), AnyArrayDType::F32 | AnyArrayDType::F64) {
            return Err(RoundCodecError::UnsupportedDtype(encoded.dtype()));
        }

        Ok(decoded.assign(&encoded)?)
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
    /// [`RoundCodec`] cannot decode into the provided array
    #[error("Round cannot decode into the provided array")]
    MismatchedDecodeIntoArray {
        /// The source of the error
        #[from]
        source: AnyArrayAssignError,
    },
}

#[must_use]
/// Rounds the input `data` using `c = round(x / precision) * precision`
pub fn round<T: Float, S: Data<Elem = T>, D: Dimension>(
    data: ArrayBase<S, D>,
    precision: Positive<T>,
) -> Array<T, D> {
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
