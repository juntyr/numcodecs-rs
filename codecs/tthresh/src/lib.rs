//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.82.0-blue
//! [repo]: https://github.com/juntyr/numcodecs-rs
//!
//! [Latest Version]: https://img.shields.io/crates/v/numcodecs-tthresh
//! [crates.io]: https://crates.io/crates/numcodecs-tthresh
//!
//! [Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs-tthresh
//! [docs.rs]: https://docs.rs/numcodecs-tthresh/
//!
//! [Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/numcodecs-rs/numcodecs_tthresh
//!
//! tthresh codec implementation for the [`numcodecs`] API.

use std::borrow::Cow;

use ndarray::{Array, Array1, ArrayBase, Data, Dimension, ShapeError};
use num_traits::Float;
use numcodecs::{
    AnyArray, AnyArrayAssignError, AnyArrayDType, AnyArrayView, AnyArrayViewMut, AnyCowArray,
    Codec, StaticCodec, StaticCodecConfig, StaticCodecVersion,
};
use schemars::{json_schema, JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

#[cfg(test)]
use ::serde_json as _;

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
// serde cannot deny unknown fields because of the flatten
#[schemars(deny_unknown_fields)]
/// Codec providing compression using tthresh
pub struct TthreshCodec {
    /// tthresh error bound
    #[serde(flatten)]
    pub error_bound: TthreshErrorBound,
    /// The codec's encoding format version. Do not provide this parameter explicitly.
    #[serde(default, rename = "_version")]
    pub version: StaticCodecVersion<0, 1, 0>,
}

/// tthresh error bound
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "eb_mode")]
#[serde(deny_unknown_fields)]
pub enum TthreshErrorBound {
    /// Relative error bound
    #[serde(rename = "eps")]
    Eps {
        /// Relative error bound
        #[serde(rename = "eb_eps")]
        eps: NonNegative<f64>,
    },
    /// Root mean square error bound
    #[serde(rename = "rmse")]
    RMSE {
        /// Peak signal to noise ratio error bound
        #[serde(rename = "eb_rmse")]
        rmse: NonNegative<f64>,
    },
    /// Peak signal-to-noise ratio error bound
    #[serde(rename = "psnr")]
    PSNR {
        /// Peak signal to noise ratio error bound
        #[serde(rename = "eb_psnr")]
        psnr: NonNegative<f64>,
    },
}

impl Codec for TthreshCodec {
    type Error = TthreshCodecError;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        match data {
            AnyCowArray::U8(data) => Ok(AnyArray::U8(
                Array1::from(compress(data, &self.error_bound)?).into_dyn(),
            )),
            AnyCowArray::U16(data) => Ok(AnyArray::U8(
                Array1::from(compress(data, &self.error_bound)?).into_dyn(),
            )),
            AnyCowArray::I32(data) => Ok(AnyArray::U8(
                Array1::from(compress(data, &self.error_bound)?).into_dyn(),
            )),
            AnyCowArray::F32(data) => Ok(AnyArray::U8(
                Array1::from(compress(data, &self.error_bound)?).into_dyn(),
            )),
            AnyCowArray::F64(data) => Ok(AnyArray::U8(
                Array1::from(compress(data, &self.error_bound)?).into_dyn(),
            )),
            encoded => Err(TthreshCodecError::UnsupportedDtype(encoded.dtype())),
        }
    }

    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
        let AnyCowArray::U8(encoded) = encoded else {
            return Err(TthreshCodecError::EncodedDataNotBytes {
                dtype: encoded.dtype(),
            });
        };

        if !matches!(encoded.shape(), [_]) {
            return Err(TthreshCodecError::EncodedDataNotOneDimensional {
                shape: encoded.shape().to_vec(),
            });
        }

        decompress(&AnyCowArray::U8(encoded).as_bytes())
    }

    fn decode_into(
        &self,
        encoded: AnyArrayView,
        mut decoded: AnyArrayViewMut,
    ) -> Result<(), Self::Error> {
        let decoded_in = self.decode(encoded.cow())?;

        Ok(decoded.assign(&decoded_in)?)
    }
}

impl StaticCodec for TthreshCodec {
    const CODEC_ID: &'static str = "tthresh.rs";

    type Config<'de> = Self;

    fn from_config(config: Self::Config<'_>) -> Self {
        config
    }

    fn get_config(&self) -> StaticCodecConfig<Self> {
        StaticCodecConfig::from(self)
    }
}

#[derive(Debug, Error)]
/// Errors that may occur when applying the [`TthreshCodec`].
pub enum TthreshCodecError {
    /// [`TthreshCodec`] does not support the dtype
    #[error("Tthresh does not support the dtype {0}")]
    UnsupportedDtype(AnyArrayDType),
    /// [`TthreshCodec`] failed to encode the data
    #[error("Tthresh failed to encode the data")]
    TthreshEncodeFailed {
        /// Opaque source error
        source: TthreshCodingError,
    },
    /// [`TthreshCodec`] can only decode one-dimensional byte arrays but received
    /// an array of a different dtype
    #[error(
        "Tthresh can only decode one-dimensional byte arrays but received an array of dtype {dtype}"
    )]
    EncodedDataNotBytes {
        /// The unexpected dtype of the encoded array
        dtype: AnyArrayDType,
    },
    /// [`TthreshCodec`] can only decode one-dimensional byte arrays but received
    /// an array of a different shape
    #[error("Tthresh can only decode one-dimensional byte arrays but received a byte array of shape {shape:?}")]
    EncodedDataNotOneDimensional {
        /// The unexpected shape of the encoded array
        shape: Vec<usize>,
    },
    /// [`TthreshCodec`] failed to decode the data
    #[error("Tthresh failed to decode the data")]
    TthreshDecodeFailed {
        /// Opaque source error
        source: TthreshCodingError,
    },
    /// [`TthreshCodec`] decoded an invalid array shape header which does not fit
    /// the decoded data
    #[error("Tthresh decoded an invalid array shape header which does not fit the decoded data")]
    DecodeInvalidShapeHeader {
        /// Source error
        #[from]
        source: ShapeError,
    },
    /// [`TthreshCodec`] cannot decode into the provided array
    #[error("Tthresh cannot decode into the provided array")]
    MismatchedDecodeIntoArray {
        /// The source of the error
        #[from]
        source: AnyArrayAssignError,
    },
}

#[derive(Debug, Error)]
#[error(transparent)]
/// Opaque error for when encoding or decoding with tthresh fails
pub struct TthreshCodingError(tthresh::Error);

#[expect(clippy::needless_pass_by_value)]
/// Compresses the input `data` array using tthresh with the provided
/// `error_bound`.
///
/// # Errors
///
/// Errors with
/// - [`TthreshCodecError::TthreshEncodeFailed`] if encoding failed with an opaque error
pub fn compress<T: TthreshElement, S: Data<Elem = T>, D: Dimension>(
    data: ArrayBase<S, D>,
    error_bound: &TthreshErrorBound,
) -> Result<Vec<u8>, TthreshCodecError> {
    let data_cow = match data.as_slice() {
        Some(data) => Cow::Borrowed(data),
        None => Cow::Owned(data.iter().copied().collect()),
    };

    let compressed = tthresh::compress(
        &data_cow,
        data.shape(),
        match error_bound {
            TthreshErrorBound::Eps { eps } => tthresh::ErrorBound::Eps(eps.0),
            TthreshErrorBound::RMSE { rmse } => tthresh::ErrorBound::RMSE(rmse.0),
            TthreshErrorBound::PSNR { psnr } => tthresh::ErrorBound::PSNR(psnr.0),
        },
        false,
        false,
    )
    .map_err(|err| TthreshCodecError::TthreshEncodeFailed {
        source: TthreshCodingError(err),
    })?;

    Ok(compressed)
}

/// Decompresses the `encoded` data into an array.
///
/// # Errors
///
/// Errors with
/// - [`TthreshCodecError::TthreshDecodeFailed`] if decoding failed with an opaque error
pub fn decompress(encoded: &[u8]) -> Result<AnyArray, TthreshCodecError> {
    let (decompressed, shape) = tthresh::decompress(encoded, false, false).map_err(|err| {
        TthreshCodecError::TthreshDecodeFailed {
            source: TthreshCodingError(err),
        }
    })?;

    let decoded = match decompressed {
        tthresh::Buffer::U8(decompressed) => {
            AnyArray::U8(Array::from_shape_vec(shape, decompressed)?.into_dyn())
        }
        tthresh::Buffer::U16(decompressed) => {
            AnyArray::U16(Array::from_shape_vec(shape, decompressed)?.into_dyn())
        }
        tthresh::Buffer::I32(decompressed) => {
            AnyArray::I32(Array::from_shape_vec(shape, decompressed)?.into_dyn())
        }
        tthresh::Buffer::F32(decompressed) => {
            AnyArray::F32(Array::from_shape_vec(shape, decompressed)?.into_dyn())
        }
        tthresh::Buffer::F64(decompressed) => {
            AnyArray::F64(Array::from_shape_vec(shape, decompressed)?.into_dyn())
        }
    };

    Ok(decoded)
}

/// Array element types which can be compressed with tthresh.
pub trait TthreshElement: Copy + tthresh::Element {}

impl TthreshElement for u8 {}
impl TthreshElement for u16 {}
impl TthreshElement for i32 {}
impl TthreshElement for f32 {}
impl TthreshElement for f64 {}

#[expect(clippy::derive_partial_eq_without_eq)] // floats are not Eq
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd, Hash)]
/// Non-negative floating point number
pub struct NonNegative<T: Float>(T);

impl Serialize for NonNegative<f64> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_f64(self.0)
    }
}

impl<'de> Deserialize<'de> for NonNegative<f64> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let x = f64::deserialize(deserializer)?;

        if x >= 0.0 {
            Ok(Self(x))
        } else {
            Err(serde::de::Error::invalid_value(
                serde::de::Unexpected::Float(x),
                &"a non-negative value",
            ))
        }
    }
}

impl JsonSchema for NonNegative<f64> {
    fn schema_name() -> Cow<'static, str> {
        Cow::Borrowed("NonNegativeF64")
    }

    fn schema_id() -> Cow<'static, str> {
        Cow::Borrowed(concat!(module_path!(), "::", "NonNegative<f64>"))
    }

    fn json_schema(_gen: &mut SchemaGenerator) -> Schema {
        json_schema!({
            "type": "number",
            "minimum": 0.0
        })
    }
}
