//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.82.0-blue
//! [repo]: https://github.com/juntyr/numcodecs-rs
//!
//! [Latest Version]: https://img.shields.io/crates/v/numcodecs-sz3
//! [crates.io]: https://crates.io/crates/numcodecs-sz3
//!
//! [Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs-sz3
//! [docs.rs]: https://docs.rs/numcodecs-sz3/
//!
//! [Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/numcodecs-rs/numcodecs_sz3
//!
//! SZ3 codec implementation for the [`numcodecs`] API.

#![allow(clippy::multiple_crate_versions)] // embedded-io

use std::{borrow::Cow, fmt};

use ndarray::{Array, Array1, ArrayBase, Data, Dimension, ShapeError};
use numcodecs::{
    AnyArray, AnyArrayAssignError, AnyArrayDType, AnyArrayView, AnyArrayViewMut, AnyCowArray,
    Codec, StaticCodec, StaticCodecConfig,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;

// Only included to explicitly enable the `no_wasm_shim` feature for
// sz3-sys/Sz3-sys
use ::zstd_sys as _;

#[cfg(test)]
use ::serde_json as _;

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
// serde cannot deny unknown fields because of the flatten
#[schemars(deny_unknown_fields)]
/// Codec providing compression using SZ3
pub struct Sz3Codec {
    /// Predictor
    #[serde(default = "default_predictor")]
    pub predictor: Option<Sz3Predictor>,
    /// SZ3 error bound
    #[serde(flatten)]
    pub error_bound: Sz3ErrorBound,
}

/// SZ3 error bound
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "eb_mode")]
#[serde(deny_unknown_fields)]
pub enum Sz3ErrorBound {
    /// Errors are bounded by *both* the absolute and relative error, i.e. by
    /// whichever bound is stricter
    #[serde(rename = "abs-and-rel")]
    AbsoluteAndRelative {
        /// Absolute error bound
        #[serde(rename = "eb_abs")]
        abs: f64,
        /// Relative error bound
        #[serde(rename = "eb_rel")]
        rel: f64,
    },
    /// Errors are bounded by *either* the absolute or relative error, i.e. by
    /// whichever bound is weaker
    #[serde(rename = "abs-or-rel")]
    AbsoluteOrRelative {
        /// Absolute error bound
        #[serde(rename = "eb_abs")]
        abs: f64,
        /// Relative error bound
        #[serde(rename = "eb_rel")]
        rel: f64,
    },
    /// Absolute error bound
    #[serde(rename = "abs")]
    Absolute {
        /// Absolute error bound
        #[serde(rename = "eb_abs")]
        abs: f64,
    },
    /// Relative error bound
    #[serde(rename = "rel")]
    Relative {
        /// Relative error bound
        #[serde(rename = "eb_rel")]
        rel: f64,
    },
    /// Peak signal to noise ratio error bound
    #[serde(rename = "psnr")]
    PS2NR {
        /// Peak signal to noise ratio error bound
        #[serde(rename = "eb_psnr")]
        psnr: f64,
    },
    /// Peak L2 norm error bound
    #[serde(rename = "l2")]
    L2Norm {
        /// Peak L2 norm error bound
        #[serde(rename = "eb_l2")]
        l2: f64,
    },
}

/// SZ3 predictor
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub enum Sz3Predictor {
    /// Linear interpolation
    #[serde(rename = "linear-interpolation")]
    LinearInterpolation,
    /// Cubic interpolation
    #[serde(rename = "cubic-interpolation")]
    CubicInterpolation,
    /// Linear interpolation + Lorenzo predictor
    #[serde(rename = "linear-interpolation-lorenzo")]
    LinearInterpolationLorenzo,
    /// Cubic interpolation + Lorenzo predictor
    #[serde(rename = "cubic-interpolation-lorenzo")]
    CubicInterpolationLorenzo,
    /// 1st order regression
    #[serde(rename = "regression")]
    Regression,
    /// 2nd order regression
    #[serde(rename = "regression2")]
    RegressionSecondOrder,
    /// 1st+2nd order regression
    #[serde(rename = "regression-regression2")]
    RegressionFirstSecondOrder,
    /// 2nd order Lorenzo predictor
    #[serde(rename = "lorenzo2")]
    LorenzoSecondOrder,
    /// 2nd order Lorenzo predictor + 2nd order regression
    #[serde(rename = "lorenzo2-regression2")]
    LorenzoSecondOrderRegressionSecondOrder,
    /// 2nd order Lorenzo predictor + 1st order regression
    #[serde(rename = "lorenzo2-regression")]
    LorenzoSecondOrderRegression,
    /// 2nd order Lorenzo predictor + 1st order regression
    #[serde(rename = "lorenzo2-regression-regression2")]
    LorenzoSecondOrderRegressionFirstSecondOrder,
    /// 1st order Lorenzo predictor
    #[serde(rename = "lorenzo")]
    Lorenzo,
    /// 1st order Lorenzo predictor + 2nd order regression
    #[serde(rename = "lorenzo-regression2")]
    LorenzoRegressionSecondOrder,
    /// 1st order Lorenzo predictor + 1st order regression
    #[serde(rename = "lorenzo-regression")]
    LorenzoRegression,
    /// 1st order Lorenzo predictor + 1st and 2nd order regression
    #[serde(rename = "lorenzo-regression-regression2")]
    LorenzoRegressionFirstSecondOrder,
    /// 1st+2nd order Lorenzo predictor
    #[serde(rename = "lorenzo-lorenzo2")]
    LorenzoFirstSecondOrder,
    /// 1st+2nd order Lorenzo predictor + 2nd order regression
    #[serde(rename = "lorenzo-lorenzo2-regression2")]
    LorenzoFirstSecondOrderRegressionSecondOrder,
    /// 1st+2nd order Lorenzo predictor + 1st order regression
    #[serde(rename = "lorenzo-lorenzo2-regression")]
    LorenzoFirstSecondOrderRegression,
    /// 1st+2nd order Lorenzo predictor + 1st+2nd order regression
    #[serde(rename = "lorenzo-lorenzo2-regression-regression2")]
    LorenzoFirstSecondOrderRegressionFirstSecondOrder,
}

#[expect(clippy::unnecessary_wraps)]
const fn default_predictor() -> Option<Sz3Predictor> {
    Some(Sz3Predictor::CubicInterpolationLorenzo)
}

impl Codec for Sz3Codec {
    type Error = Sz3CodecError;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        match data {
            AnyCowArray::I32(data) => Ok(AnyArray::U8(
                Array1::from(compress(data, self.predictor.as_ref(), &self.error_bound)?)
                    .into_dyn(),
            )),
            AnyCowArray::I64(data) => Ok(AnyArray::U8(
                Array1::from(compress(data, self.predictor.as_ref(), &self.error_bound)?)
                    .into_dyn(),
            )),
            AnyCowArray::F32(data) => Ok(AnyArray::U8(
                Array1::from(compress(data, self.predictor.as_ref(), &self.error_bound)?)
                    .into_dyn(),
            )),
            AnyCowArray::F64(data) => Ok(AnyArray::U8(
                Array1::from(compress(data, self.predictor.as_ref(), &self.error_bound)?)
                    .into_dyn(),
            )),
            encoded => Err(Sz3CodecError::UnsupportedDtype(encoded.dtype())),
        }
    }

    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
        let AnyCowArray::U8(encoded) = encoded else {
            return Err(Sz3CodecError::EncodedDataNotBytes {
                dtype: encoded.dtype(),
            });
        };

        if !matches!(encoded.shape(), [_]) {
            return Err(Sz3CodecError::EncodedDataNotOneDimensional {
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

impl StaticCodec for Sz3Codec {
    const CODEC_ID: &'static str = "sz3";

    type Config<'de> = Self;

    fn from_config(config: Self::Config<'_>) -> Self {
        config
    }

    fn get_config(&self) -> StaticCodecConfig<Self> {
        StaticCodecConfig::from(self)
    }
}

#[derive(Debug, Error)]
/// Errors that may occur when applying the [`Sz3Codec`].
pub enum Sz3CodecError {
    /// [`Sz3Codec`] does not support the dtype
    #[error("Sz3 does not support the dtype {0}")]
    UnsupportedDtype(AnyArrayDType),
    /// [`Sz3Codec`] failed to encode the header
    #[error("Sz3 failed to encode the header")]
    HeaderEncodeFailed {
        /// Opaque source error
        source: Sz3HeaderError,
    },
    /// [`Sz3Codec`] cannot encode an array of `shape`
    #[error("Sz3 cannot encode an array of shape {shape:?}")]
    InvalidEncodeShape {
        /// Opaque source error
        source: Sz3CodingError,
        /// The invalid shape of the encoded array
        shape: Vec<usize>,
    },
    /// [`Sz3Codec`] failed to encode the data
    #[error("Sz3 failed to encode the data")]
    Sz3EncodeFailed {
        /// Opaque source error
        source: Sz3CodingError,
    },
    /// [`Sz3Codec`] can only decode one-dimensional byte arrays but received
    /// an array of a different dtype
    #[error(
        "Sz3 can only decode one-dimensional byte arrays but received an array of dtype {dtype}"
    )]
    EncodedDataNotBytes {
        /// The unexpected dtype of the encoded array
        dtype: AnyArrayDType,
    },
    /// [`Sz3Codec`] can only decode one-dimensional byte arrays but received
    /// an array of a different shape
    #[error("Sz3 can only decode one-dimensional byte arrays but received a byte array of shape {shape:?}")]
    EncodedDataNotOneDimensional {
        /// The unexpected shape of the encoded array
        shape: Vec<usize>,
    },
    /// [`Sz3Codec`] failed to decode the header
    #[error("Sz3 failed to decode the header")]
    HeaderDecodeFailed {
        /// Opaque source error
        source: Sz3HeaderError,
    },
    /// [`Sz3Codec`] decoded an invalid array shape header which does not fit
    /// the decoded data
    #[error("Sz3 decoded an invalid array shape header which does not fit the decoded data")]
    DecodeInvalidShapeHeader {
        /// Source error
        #[from]
        source: ShapeError,
    },
    /// [`Sz3Codec`] cannot decode into the provided array
    #[error("Sz3 cannot decode into the provided array")]
    MismatchedDecodeIntoArray {
        /// The source of the error
        #[from]
        source: AnyArrayAssignError,
    },
}

#[derive(Debug, Error)]
#[error(transparent)]
/// Opaque error for when encoding or decoding the header fails
pub struct Sz3HeaderError(postcard::Error);

#[derive(Debug, Error)]
#[error(transparent)]
/// Opaque error for when encoding or decoding with SZ3 fails
pub struct Sz3CodingError(sz3::SZ3Error);

#[expect(clippy::needless_pass_by_value, clippy::too_many_lines)]
/// Compresses the input `data` array using SZ3, which consists of an optional
/// `predictor`, an `error_bound`, an optional `encoder`, and an optional
/// `lossless` compressor.
///
/// # Errors
///
/// Errors with
/// - [`Sz3CodecError::HeaderEncodeFailed`] if encoding the header failed
/// - [`Sz3CodecError::InvalidEncodeShape`] if the array shape is invalid
/// - [`Sz3CodecError::Sz3EncodeFailed`] if encoding failed with an opaque error
pub fn compress<T: Sz3Element, S: Data<Elem = T>, D: Dimension>(
    data: ArrayBase<S, D>,
    predictor: Option<&Sz3Predictor>,
    error_bound: &Sz3ErrorBound,
) -> Result<Vec<u8>, Sz3CodecError> {
    let mut encoded_bytes = postcard::to_extend(
        &CompressionHeader {
            dtype: <T as Sz3Element>::DTYPE,
            shape: Cow::Borrowed(data.shape()),
        },
        Vec::new(),
    )
    .map_err(|err| Sz3CodecError::HeaderEncodeFailed {
        source: Sz3HeaderError(err),
    })?;

    // sz3::DimensionedDataBuilder cannot handle zero-length dimensions
    if data.is_empty() {
        return Ok(encoded_bytes);
    }

    #[expect(clippy::option_if_let_else)]
    let data_cow = if let Some(data) = data.as_slice() {
        Cow::Borrowed(data)
    } else {
        Cow::Owned(data.iter().copied().collect())
    };
    let mut builder = sz3::DimensionedData::build(&data_cow);

    for length in data.shape() {
        // Sz3 ignores dimensions of length 1 and panics on length zero
        // Since they carry no information for Sz3 and we already encode them
        //  in our custom header, we just skip them here
        if *length > 1 {
            builder = builder
                .dim(*length)
                .map_err(|err| Sz3CodecError::InvalidEncodeShape {
                    source: Sz3CodingError(err),
                    shape: data.shape().to_vec(),
                })?;
        }
    }

    if data.len() == 1 {
        // If there is only one element, all dimensions will have been skipped,
        //  so we explicitly encode one dimension of size 1 here
        builder = builder
            .dim(1)
            .map_err(|err| Sz3CodecError::InvalidEncodeShape {
                source: Sz3CodingError(err),
                shape: data.shape().to_vec(),
            })?;
    }

    let data = builder
        .finish()
        .map_err(|err| Sz3CodecError::InvalidEncodeShape {
            source: Sz3CodingError(err),
            shape: data.shape().to_vec(),
        })?;

    // configure the error bound
    let error_bound = match error_bound {
        Sz3ErrorBound::AbsoluteAndRelative { abs, rel } => sz3::ErrorBound::AbsoluteAndRelative {
            absolute_bound: *abs,
            relative_bound: *rel,
        },
        Sz3ErrorBound::AbsoluteOrRelative { abs, rel } => sz3::ErrorBound::AbsoluteOrRelative {
            absolute_bound: *abs,
            relative_bound: *rel,
        },
        Sz3ErrorBound::Absolute { abs } => sz3::ErrorBound::Absolute(*abs),
        Sz3ErrorBound::Relative { rel } => sz3::ErrorBound::Relative(*rel),
        Sz3ErrorBound::PS2NR { psnr } => sz3::ErrorBound::PSNR(*psnr),
        Sz3ErrorBound::L2Norm { l2 } => sz3::ErrorBound::L2Norm(*l2),
    };
    let mut config = sz3::Config::new(error_bound);

    // configure the interpolation mode, if necessary
    let interpolation = match predictor {
        Some(Sz3Predictor::LinearInterpolation | Sz3Predictor::LinearInterpolationLorenzo) => {
            Some(sz3::InterpolationAlgorithm::Linear)
        }
        Some(Sz3Predictor::CubicInterpolation | Sz3Predictor::CubicInterpolationLorenzo) => {
            Some(sz3::InterpolationAlgorithm::Cubic)
        }
        Some(
            Sz3Predictor::Regression
            | Sz3Predictor::RegressionSecondOrder
            | Sz3Predictor::RegressionFirstSecondOrder
            | Sz3Predictor::LorenzoSecondOrder
            | Sz3Predictor::LorenzoSecondOrderRegressionSecondOrder
            | Sz3Predictor::LorenzoSecondOrderRegression
            | Sz3Predictor::LorenzoSecondOrderRegressionFirstSecondOrder
            | Sz3Predictor::Lorenzo
            | Sz3Predictor::LorenzoRegressionSecondOrder
            | Sz3Predictor::LorenzoRegression
            | Sz3Predictor::LorenzoRegressionFirstSecondOrder
            | Sz3Predictor::LorenzoFirstSecondOrder
            | Sz3Predictor::LorenzoFirstSecondOrderRegressionSecondOrder
            | Sz3Predictor::LorenzoFirstSecondOrderRegression
            | Sz3Predictor::LorenzoFirstSecondOrderRegressionFirstSecondOrder,
        )
        | None => None,
    };
    if let Some(interpolation) = interpolation {
        config = config.interpolation_algorithm(interpolation);
    }

    // configure the predictor (compression algorithm)
    let predictor = match predictor {
        Some(Sz3Predictor::LinearInterpolation | Sz3Predictor::CubicInterpolation) => {
            sz3::CompressionAlgorithm::Interpolation
        }
        Some(
            Sz3Predictor::LinearInterpolationLorenzo | Sz3Predictor::CubicInterpolationLorenzo,
        ) => sz3::CompressionAlgorithm::InterpolationLorenzo,
        Some(Sz3Predictor::RegressionSecondOrder) => sz3::CompressionAlgorithm::LorenzoRegression {
            lorenzo: false,
            lorenzo_second_order: false,
            regression: false,
            regression_second_order: true,
            prediction_dimension: None,
        },
        Some(Sz3Predictor::Regression) => sz3::CompressionAlgorithm::LorenzoRegression {
            lorenzo: false,
            lorenzo_second_order: false,
            regression: true,
            regression_second_order: false,
            prediction_dimension: None,
        },
        Some(Sz3Predictor::RegressionFirstSecondOrder) => {
            sz3::CompressionAlgorithm::LorenzoRegression {
                lorenzo: false,
                lorenzo_second_order: false,
                regression: true,
                regression_second_order: true,
                prediction_dimension: None,
            }
        }
        Some(Sz3Predictor::LorenzoSecondOrder) => sz3::CompressionAlgorithm::LorenzoRegression {
            lorenzo: false,
            lorenzo_second_order: true,
            regression: false,
            regression_second_order: false,
            prediction_dimension: None,
        },
        Some(Sz3Predictor::LorenzoSecondOrderRegressionSecondOrder) => {
            sz3::CompressionAlgorithm::LorenzoRegression {
                lorenzo: false,
                lorenzo_second_order: true,
                regression: false,
                regression_second_order: true,
                prediction_dimension: None,
            }
        }
        Some(Sz3Predictor::LorenzoSecondOrderRegression) => {
            sz3::CompressionAlgorithm::LorenzoRegression {
                lorenzo: false,
                lorenzo_second_order: true,
                regression: true,
                regression_second_order: false,
                prediction_dimension: None,
            }
        }
        Some(Sz3Predictor::LorenzoSecondOrderRegressionFirstSecondOrder) => {
            sz3::CompressionAlgorithm::LorenzoRegression {
                lorenzo: false,
                lorenzo_second_order: true,
                regression: true,
                regression_second_order: true,
                prediction_dimension: None,
            }
        }
        Some(Sz3Predictor::Lorenzo) => sz3::CompressionAlgorithm::LorenzoRegression {
            lorenzo: true,
            lorenzo_second_order: false,
            regression: false,
            regression_second_order: false,
            prediction_dimension: None,
        },
        Some(Sz3Predictor::LorenzoRegressionSecondOrder) => {
            sz3::CompressionAlgorithm::LorenzoRegression {
                lorenzo: true,
                lorenzo_second_order: false,
                regression: false,
                regression_second_order: true,
                prediction_dimension: None,
            }
        }
        Some(Sz3Predictor::LorenzoRegression) => sz3::CompressionAlgorithm::LorenzoRegression {
            lorenzo: true,
            lorenzo_second_order: false,
            regression: true,
            regression_second_order: false,
            prediction_dimension: None,
        },
        Some(Sz3Predictor::LorenzoRegressionFirstSecondOrder) => {
            sz3::CompressionAlgorithm::LorenzoRegression {
                lorenzo: true,
                lorenzo_second_order: false,
                regression: true,
                regression_second_order: true,
                prediction_dimension: None,
            }
        }
        Some(Sz3Predictor::LorenzoFirstSecondOrder) => {
            sz3::CompressionAlgorithm::LorenzoRegression {
                lorenzo: true,
                lorenzo_second_order: true,
                regression: false,
                regression_second_order: false,
                prediction_dimension: None,
            }
        }
        Some(Sz3Predictor::LorenzoFirstSecondOrderRegressionSecondOrder) => {
            sz3::CompressionAlgorithm::LorenzoRegression {
                lorenzo: true,
                lorenzo_second_order: true,
                regression: false,
                regression_second_order: true,
                prediction_dimension: None,
            }
        }
        Some(Sz3Predictor::LorenzoFirstSecondOrderRegression) => {
            sz3::CompressionAlgorithm::LorenzoRegression {
                lorenzo: true,
                lorenzo_second_order: true,
                regression: true,
                regression_second_order: false,
                prediction_dimension: None,
            }
        }
        Some(Sz3Predictor::LorenzoFirstSecondOrderRegressionFirstSecondOrder) => {
            sz3::CompressionAlgorithm::LorenzoRegression {
                lorenzo: true,
                lorenzo_second_order: true,
                regression: true,
                regression_second_order: true,
                prediction_dimension: None,
            }
        }
        None => sz3::CompressionAlgorithm::NoPrediction,
    };
    config = config.compression_algorithm(predictor);

    // TODO: avoid extra allocation here
    let compressed = sz3::compress_with_config(&data, &config).map_err(|err| {
        Sz3CodecError::Sz3EncodeFailed {
            source: Sz3CodingError(err),
        }
    })?;
    encoded_bytes.extend_from_slice(&compressed);

    Ok(encoded_bytes)
}

/// Decompresses the `encoded` data into an array.
///
/// # Errors
///
/// Errors with
/// - [`Sz3CodecError::HeaderDecodeFailed`] if decoding the header failed
pub fn decompress(encoded: &[u8]) -> Result<AnyArray, Sz3CodecError> {
    let (header, data) =
        postcard::take_from_bytes::<CompressionHeader>(encoded).map_err(|err| {
            Sz3CodecError::HeaderDecodeFailed {
                source: Sz3HeaderError(err),
            }
        })?;

    let decoded = if header.shape.iter().copied().product::<usize>() == 0 {
        match header.dtype {
            Sz3DType::I32 => {
                AnyArray::I32(Array::from_shape_vec(&*header.shape, Vec::new())?.into_dyn())
            }
            Sz3DType::I64 => {
                AnyArray::I64(Array::from_shape_vec(&*header.shape, Vec::new())?.into_dyn())
            }
            Sz3DType::F32 => {
                AnyArray::F32(Array::from_shape_vec(&*header.shape, Vec::new())?.into_dyn())
            }
            Sz3DType::F64 => {
                AnyArray::F64(Array::from_shape_vec(&*header.shape, Vec::new())?.into_dyn())
            }
        }
    } else {
        // TODO: avoid extra allocation here
        match header.dtype {
            Sz3DType::I32 => AnyArray::I32(Array::from_shape_vec(
                &*header.shape,
                Vec::from(sz3::decompress(data).1.data()),
            )?),
            Sz3DType::I64 => AnyArray::I64(Array::from_shape_vec(
                &*header.shape,
                Vec::from(sz3::decompress(data).1.data()),
            )?),
            Sz3DType::F32 => AnyArray::F32(Array::from_shape_vec(
                &*header.shape,
                Vec::from(sz3::decompress(data).1.data()),
            )?),
            Sz3DType::F64 => AnyArray::F64(Array::from_shape_vec(
                &*header.shape,
                Vec::from(sz3::decompress(data).1.data()),
            )?),
        }
    };

    Ok(decoded)
}

/// Array element types which can be compressed with SZ3.
pub trait Sz3Element: Copy + sz3::SZ3Compressible {
    /// The dtype representation of the type
    const DTYPE: Sz3DType;
}

impl Sz3Element for i32 {
    const DTYPE: Sz3DType = Sz3DType::I32;
}

impl Sz3Element for i64 {
    const DTYPE: Sz3DType = Sz3DType::I64;
}

impl Sz3Element for f32 {
    const DTYPE: Sz3DType = Sz3DType::F32;
}

impl Sz3Element for f64 {
    const DTYPE: Sz3DType = Sz3DType::F64;
}

#[derive(Serialize, Deserialize)]
struct CompressionHeader<'a> {
    dtype: Sz3DType,
    #[serde(borrow)]
    shape: Cow<'a, [usize]>,
}

/// Dtypes that SZ3 can compress and decompress
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
#[expect(missing_docs)]
pub enum Sz3DType {
    #[serde(rename = "i32", alias = "int32")]
    I32,
    #[serde(rename = "i64", alias = "int64")]
    I64,
    #[serde(rename = "f32", alias = "float32")]
    F32,
    #[serde(rename = "f64", alias = "float64")]
    F64,
}

impl fmt::Display for Sz3DType {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.write_str(match self {
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::F32 => "f32",
            Self::F64 => "f64",
        })
    }
}

#[cfg(test)]
mod tests {
    use ndarray::ArrayView1;

    use super::*;

    #[test]
    fn zero_length() -> Result<(), Sz3CodecError> {
        let encoded = compress(
            Array::<f32, _>::from_shape_vec([1, 27, 0].as_slice(), vec![])?,
            default_predictor().as_ref(),
            &Sz3ErrorBound::L2Norm { l2: 27.0 },
        )?;
        let decoded = decompress(&encoded)?;

        assert_eq!(decoded.dtype(), AnyArrayDType::F32);
        assert!(decoded.is_empty());
        assert_eq!(decoded.shape(), &[1, 27, 0]);

        Ok(())
    }

    #[test]
    fn one_dimension() -> Result<(), Sz3CodecError> {
        let data = Array::from_shape_vec([2_usize, 1, 2, 1].as_slice(), vec![1, 2, 3, 4])?;

        let encoded = compress(
            data.view(),
            default_predictor().as_ref(),
            &Sz3ErrorBound::Absolute { abs: 0.1 },
        )?;
        let decoded = decompress(&encoded)?;

        assert_eq!(decoded, AnyArray::I32(data));

        Ok(())
    }

    #[test]
    fn small_state() -> Result<(), Sz3CodecError> {
        for data in [
            &[][..],
            &[0.0],
            &[0.0, 1.0],
            &[0.0, 1.0, 0.0],
            &[0.0, 1.0, 0.0, 1.0],
        ] {
            let encoded = compress(
                ArrayView1::from(data),
                default_predictor().as_ref(),
                &Sz3ErrorBound::Absolute { abs: 0.1 },
            )?;
            let decoded = decompress(&encoded)?;

            assert_eq!(
                decoded,
                AnyArray::F64(Array1::from_vec(data.to_vec()).into_dyn())
            );
        }

        Ok(())
    }

    #[test]
    fn all_predictors() -> Result<(), Sz3CodecError> {
        let data = Array::linspace(-42.0, 42.0, 85);

        for predictor in [
            None,
            Some(Sz3Predictor::Regression),
            Some(Sz3Predictor::RegressionSecondOrder),
            Some(Sz3Predictor::RegressionFirstSecondOrder),
            Some(Sz3Predictor::LorenzoSecondOrder),
            Some(Sz3Predictor::LorenzoSecondOrderRegressionSecondOrder),
            Some(Sz3Predictor::LorenzoSecondOrderRegression),
            Some(Sz3Predictor::LorenzoSecondOrderRegressionFirstSecondOrder),
            Some(Sz3Predictor::Lorenzo),
            Some(Sz3Predictor::LorenzoRegressionSecondOrder),
            Some(Sz3Predictor::LorenzoRegression),
            Some(Sz3Predictor::LorenzoRegressionFirstSecondOrder),
            Some(Sz3Predictor::LorenzoFirstSecondOrder),
            Some(Sz3Predictor::LorenzoFirstSecondOrderRegressionSecondOrder),
            Some(Sz3Predictor::LorenzoFirstSecondOrderRegression),
            Some(Sz3Predictor::LorenzoFirstSecondOrderRegressionFirstSecondOrder),
        ] {
            let encoded = compress(
                data.view(),
                predictor.as_ref(),
                &Sz3ErrorBound::Absolute { abs: 0.1 },
            )?;
            let _decoded = decompress(&encoded)?;
        }

        Ok(())
    }
}
