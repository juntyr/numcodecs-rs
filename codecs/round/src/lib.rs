//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.82.0-blue
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

use std::borrow::Cow;

use ndarray::{Array, ArrayBase, Data, Dimension};
use num_traits::Float;
use numcodecs::{
    AnyArray, AnyArrayAssignError, AnyArrayDType, AnyArrayView, AnyArrayViewMut, AnyCowArray,
    Codec, StaticCodec, StaticCodecConfig,
};
use schemars::{json_schema, JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
/// Codec that rounds the data on encoding and passes through the input
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
            #[expect(clippy::cast_possible_truncation)]
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
}

impl StaticCodec for RoundCodec {
    const CODEC_ID: &'static str = "round";

    type Config<'de> = Self;

    fn from_config(config: Self::Config<'_>) -> Self {
        config
    }

    fn get_config(&self) -> StaticCodecConfig<Self> {
        StaticCodecConfig::from(self)
    }
}

#[expect(clippy::derive_partial_eq_without_eq)] // floats are not Eq
#[derive(Copy, Clone, PartialEq, PartialOrd, Hash)]
/// Positive floating point number
pub struct Positive<T: Float>(T);

impl Serialize for Positive<f64> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_f64(self.0)
    }
}

impl<'de> Deserialize<'de> for Positive<f64> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
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

impl JsonSchema for Positive<f64> {
    fn schema_name() -> Cow<'static, str> {
        Cow::Borrowed("PositiveF64")
    }

    fn schema_id() -> Cow<'static, str> {
        Cow::Borrowed(concat!(module_path!(), "::", "Positive<f64>"))
    }

    fn json_schema(_gen: &mut SchemaGenerator) -> Schema {
        json_schema!({
            "type": "number",
            "exclusiveMinimum": 0.0
        })
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
/// Rounds the input `data` using
/// `$c = \text{round}\left( \frac{x}{precision} \right) \cdot precision$`
pub fn round<T: Float, S: Data<Elem = T>, D: Dimension>(
    data: ArrayBase<S, D>,
    precision: Positive<T>,
) -> Array<T, D> {
    let mut encoded = data.into_owned();
    encoded.mapv_inplace(|x| (x / precision.0).round() * precision.0);
    encoded
}
