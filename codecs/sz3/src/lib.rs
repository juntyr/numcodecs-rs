//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.87.0-blue
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

use ndarray::{Array, Array1, ArrayBase, ArrayViewMut, Data, Dimension, IxDyn, ShapeError};
use numcodecs::{
    AnyArray, AnyArrayAssignError, AnyArrayDType, AnyArrayView, AnyArrayViewMut, AnyCowArray,
    ArrayDType, ArrayDataMutExt, Codec, StaticCodec, StaticCodecConfig, StaticCodecVersion,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;

// Only included to explicitly enable the `no_wasm_shim` feature for
// sz3-sys/zstd-sys
use ::zstd_sys as _;

#[cfg(test)]
use ::serde_json as _;

type Sz3CodecVersion = StaticCodecVersion<0, 2, 0>;

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
    /// The codec's encoding format version. Do not provide this parameter explicitly.
    #[serde(default, rename = "_version")]
    pub version: Sz3CodecVersion,
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
    /// Interpolation
    #[serde(rename = "interpolation")]
    Interpolation,
    /// Interpolation + Lorenzo predictor
    #[serde(rename = "interpolation-lorenzo")]
    InterpolationLorenzo,
    /// 1st order regression
    #[serde(rename = "regression")]
    Regression,
    /// 2nd order Lorenzo predictor
    #[serde(rename = "lorenzo2")]
    LorenzoSecondOrder,
    /// 2nd order Lorenzo predictor + 1st order regression
    #[serde(rename = "lorenzo2-regression")]
    LorenzoSecondOrderRegression,
    /// 1st order Lorenzo predictor
    #[serde(rename = "lorenzo")]
    Lorenzo,
    /// 1st order Lorenzo predictor + 1st order regression
    #[serde(rename = "lorenzo-regression")]
    LorenzoRegression,
    /// 1st+2nd order Lorenzo predictor
    #[serde(rename = "lorenzo-lorenzo2")]
    LorenzoFirstSecondOrder,
    /// 1st+2nd order Lorenzo predictor + 1st order regression
    #[serde(rename = "lorenzo-lorenzo2-regression")]
    LorenzoFirstSecondOrderRegression,
}

#[expect(clippy::unnecessary_wraps)]
const fn default_predictor() -> Option<Sz3Predictor> {
    Some(Sz3Predictor::InterpolationLorenzo)
}

impl Codec for Sz3Codec {
    type Error = Sz3CodecError;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        match data {
            AnyCowArray::U8(data) => Ok(AnyArray::U8(
                Array1::from(compress(data, self.predictor.as_ref(), &self.error_bound)?)
                    .into_dyn(),
            )),
            AnyCowArray::I8(data) => Ok(AnyArray::U8(
                Array1::from(compress(data, self.predictor.as_ref(), &self.error_bound)?)
                    .into_dyn(),
            )),
            AnyCowArray::U16(data) => Ok(AnyArray::U8(
                Array1::from(compress(data, self.predictor.as_ref(), &self.error_bound)?)
                    .into_dyn(),
            )),
            AnyCowArray::I16(data) => Ok(AnyArray::U8(
                Array1::from(compress(data, self.predictor.as_ref(), &self.error_bound)?)
                    .into_dyn(),
            )),
            AnyCowArray::U32(data) => Ok(AnyArray::U8(
                Array1::from(compress(data, self.predictor.as_ref(), &self.error_bound)?)
                    .into_dyn(),
            )),
            AnyCowArray::I32(data) => Ok(AnyArray::U8(
                Array1::from(compress(data, self.predictor.as_ref(), &self.error_bound)?)
                    .into_dyn(),
            )),
            AnyCowArray::U64(data) => Ok(AnyArray::U8(
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
        decoded: AnyArrayViewMut,
    ) -> Result<(), Self::Error> {
        let AnyArrayView::U8(encoded) = encoded else {
            return Err(Sz3CodecError::EncodedDataNotBytes {
                dtype: encoded.dtype(),
            });
        };

        if !matches!(encoded.shape(), [_]) {
            return Err(Sz3CodecError::EncodedDataNotOneDimensional {
                shape: encoded.shape().to_vec(),
            });
        }

        decompress_into(&AnyArrayView::U8(encoded).as_bytes(), decoded)
    }
}

impl StaticCodec for Sz3Codec {
    const CODEC_ID: &'static str = "sz3.rs";

    type Config<'de> = Self;

    fn from_config(config: Self::Config<'_>) -> Self {
        config
    }

    fn get_config(&self) -> StaticCodecConfig<'_, Self> {
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
    #[error(
        "Sz3 can only decode one-dimensional byte arrays but received a byte array of shape {shape:?}"
    )]
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
    /// [`Sz3Codec`] failed to decode the data
    #[error("Sz3 failed to decode the data")]
    Sz3DecodeFailed {
        /// Opaque source error
        source: Sz3CodingError,
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
            version: StaticCodecVersion,
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
    let data_cow = match data.as_slice() {
        Some(data) => Cow::Borrowed(data),
        None => Cow::Owned(data.iter().copied().collect()),
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

    // configure the predictor (compression algorithm)
    let predictor = match predictor {
        Some(Sz3Predictor::Interpolation) => sz3::CompressionAlgorithm::Interpolation,
        Some(Sz3Predictor::InterpolationLorenzo) => sz3::CompressionAlgorithm::InterpolationLorenzo,
        Some(Sz3Predictor::Regression) => sz3::CompressionAlgorithm::LorenzoRegression {
            lorenzo: false,
            lorenzo_second_order: false,
            regression: true,
        },
        Some(Sz3Predictor::LorenzoSecondOrder) => sz3::CompressionAlgorithm::LorenzoRegression {
            lorenzo: false,
            lorenzo_second_order: true,
            regression: false,
        },
        Some(Sz3Predictor::LorenzoSecondOrderRegression) => {
            sz3::CompressionAlgorithm::LorenzoRegression {
                lorenzo: false,
                lorenzo_second_order: true,
                regression: true,
            }
        }
        Some(Sz3Predictor::Lorenzo) => sz3::CompressionAlgorithm::LorenzoRegression {
            lorenzo: true,
            lorenzo_second_order: false,
            regression: false,
        },
        Some(Sz3Predictor::LorenzoRegression) => sz3::CompressionAlgorithm::LorenzoRegression {
            lorenzo: true,
            lorenzo_second_order: false,
            regression: true,
        },
        Some(Sz3Predictor::LorenzoFirstSecondOrder) => {
            sz3::CompressionAlgorithm::LorenzoRegression {
                lorenzo: true,
                lorenzo_second_order: true,
                regression: false,
            }
        }
        Some(Sz3Predictor::LorenzoFirstSecondOrderRegression) => {
            sz3::CompressionAlgorithm::LorenzoRegression {
                lorenzo: true,
                lorenzo_second_order: true,
                regression: true,
            }
        }
        None => sz3::CompressionAlgorithm::NoPrediction,
    };
    config = config.compression_algorithm(predictor);

    sz3::compress_into_with_config(&data, &config, &mut encoded_bytes).map_err(|err| {
        Sz3CodecError::Sz3EncodeFailed {
            source: Sz3CodingError(err),
        }
    })?;

    Ok(encoded_bytes)
}

/// Decompresses the `encoded` data into an array using SZ3.
///
/// # Errors
///
/// Errors with
/// - [`Sz3CodecError::HeaderDecodeFailed`] if decoding the header failed
/// - [`Sz3CodecError::Sz3DecodeFailed`] if decoding failed with an opaque error
pub fn decompress(encoded: &[u8]) -> Result<AnyArray, Sz3CodecError> {
    fn decompress_typed<T: Sz3Element>(
        encoded: &[u8],
        shape: &[usize],
    ) -> Result<Array<T, IxDyn>, Sz3CodecError> {
        if shape.iter().copied().any(|s| s == 0) {
            return Ok(Array::from_shape_vec(shape, Vec::new())?);
        }

        let (_config, decompressed) =
            sz3::decompress(encoded).map_err(|err| Sz3CodecError::Sz3DecodeFailed {
                source: Sz3CodingError(err),
            })?;

        Ok(Array::from_shape_vec(shape, decompressed.into_data())?)
    }

    let (header, data) =
        postcard::take_from_bytes::<CompressionHeader>(encoded).map_err(|err| {
            Sz3CodecError::HeaderDecodeFailed {
                source: Sz3HeaderError(err),
            }
        })?;

    let decoded = match header.dtype {
        Sz3DType::U8 => AnyArray::U8(decompress_typed(data, &header.shape)?),
        Sz3DType::I8 => AnyArray::I8(decompress_typed(data, &header.shape)?),
        Sz3DType::U16 => AnyArray::U16(decompress_typed(data, &header.shape)?),
        Sz3DType::I16 => AnyArray::I16(decompress_typed(data, &header.shape)?),
        Sz3DType::U32 => AnyArray::U32(decompress_typed(data, &header.shape)?),
        Sz3DType::I32 => AnyArray::I32(decompress_typed(data, &header.shape)?),
        Sz3DType::U64 => AnyArray::U64(decompress_typed(data, &header.shape)?),
        Sz3DType::I64 => AnyArray::I64(decompress_typed(data, &header.shape)?),
        Sz3DType::F32 => AnyArray::F32(decompress_typed(data, &header.shape)?),
        Sz3DType::F64 => AnyArray::F64(decompress_typed(data, &header.shape)?),
    };

    Ok(decoded)
}

/// Decompresses the `encoded` data into a `decoded` array using SZ3.
///
/// # Errors
///
/// Errors with
/// - [`Sz3CodecError::HeaderDecodeFailed`] if decoding the header failed
/// - [`Sz3CodecError::MismatchedDecodeIntoArray`] if the `decoded` array is of
///   the wrong dtype or shape
/// - [`Sz3CodecError::Sz3DecodeFailed`] if decoding failed with an opaque error
pub fn decompress_into(encoded: &[u8], decoded: AnyArrayViewMut) -> Result<(), Sz3CodecError> {
    fn decompress_into_typed<T: Sz3Element>(
        encoded: &[u8],
        mut decoded: ArrayViewMut<T, IxDyn>,
    ) -> Result<(), Sz3CodecError> {
        if decoded.is_empty() {
            return Ok(());
        }

        let decoded_shape = decoded.shape().to_vec();

        decoded.with_slice_mut(|mut decoded| {
            let decoded_len = decoded.len();

            let mut builder = sz3::DimensionedData::build_mut(&mut decoded);

            for length in &decoded_shape {
                // Sz3 ignores dimensions of length 1 and panics on length zero
                // Since they carry no information for Sz3 and we already encode them
                //  in our custom header, we just skip them here
                if *length > 1 {
                    builder = builder
                        .dim(*length)
                        // FIXME: different error code
                        .map_err(|err| Sz3CodecError::InvalidEncodeShape {
                            source: Sz3CodingError(err),
                            shape: decoded_shape.clone(),
                        })?;
                }
            }

            if decoded_len == 1 {
                // If there is only one element, all dimensions will have been skipped,
                //  so we explicitly encode one dimension of size 1 here
                builder = builder
                    .dim(1)
                    // FIXME: different error code
                    .map_err(|err| Sz3CodecError::InvalidEncodeShape {
                        source: Sz3CodingError(err),
                        shape: decoded_shape.clone(),
                    })?;
            }

            let mut decoded = builder
                .finish()
                // FIXME: different error code
                .map_err(|err| Sz3CodecError::InvalidEncodeShape {
                    source: Sz3CodingError(err),
                    shape: decoded_shape,
                })?;

            sz3::decompress_into_dimensioned(encoded, &mut decoded).map_err(|err| {
                Sz3CodecError::Sz3DecodeFailed {
                    source: Sz3CodingError(err),
                }
            })
        })?;

        Ok(())
    }

    let (header, data) =
        postcard::take_from_bytes::<CompressionHeader>(encoded).map_err(|err| {
            Sz3CodecError::HeaderDecodeFailed {
                source: Sz3HeaderError(err),
            }
        })?;

    if decoded.shape() != &*header.shape {
        return Err(Sz3CodecError::MismatchedDecodeIntoArray {
            source: AnyArrayAssignError::ShapeMismatch {
                src: header.shape.into_owned(),
                dst: decoded.shape().to_vec(),
            },
        });
    }

    match (decoded, header.dtype) {
        (AnyArrayViewMut::U8(decoded), Sz3DType::U8) => decompress_into_typed(data, decoded),
        (AnyArrayViewMut::I8(decoded), Sz3DType::I8) => decompress_into_typed(data, decoded),
        (AnyArrayViewMut::U16(decoded), Sz3DType::U16) => decompress_into_typed(data, decoded),
        (AnyArrayViewMut::I16(decoded), Sz3DType::I16) => decompress_into_typed(data, decoded),
        (AnyArrayViewMut::U32(decoded), Sz3DType::U32) => decompress_into_typed(data, decoded),
        (AnyArrayViewMut::I32(decoded), Sz3DType::I32) => decompress_into_typed(data, decoded),
        (AnyArrayViewMut::U64(decoded), Sz3DType::U64) => decompress_into_typed(data, decoded),
        (AnyArrayViewMut::I64(decoded), Sz3DType::I64) => decompress_into_typed(data, decoded),
        (AnyArrayViewMut::F32(decoded), Sz3DType::F32) => decompress_into_typed(data, decoded),
        (AnyArrayViewMut::F64(decoded), Sz3DType::F64) => decompress_into_typed(data, decoded),
        (decoded, dtype) => Err(Sz3CodecError::MismatchedDecodeIntoArray {
            source: AnyArrayAssignError::DTypeMismatch {
                src: dtype.into_dtype(),
                dst: decoded.dtype(),
            },
        }),
    }
}

/// Array element types which can be compressed with SZ3.
pub trait Sz3Element: Copy + sz3::SZ3Compressible + ArrayDType {
    /// The dtype representation of the type
    const DTYPE: Sz3DType;
}

impl Sz3Element for u8 {
    const DTYPE: Sz3DType = Sz3DType::U8;
}

impl Sz3Element for i8 {
    const DTYPE: Sz3DType = Sz3DType::I8;
}

impl Sz3Element for u16 {
    const DTYPE: Sz3DType = Sz3DType::U16;
}

impl Sz3Element for i16 {
    const DTYPE: Sz3DType = Sz3DType::I16;
}

impl Sz3Element for u32 {
    const DTYPE: Sz3DType = Sz3DType::U32;
}

impl Sz3Element for i32 {
    const DTYPE: Sz3DType = Sz3DType::I32;
}

impl Sz3Element for u64 {
    const DTYPE: Sz3DType = Sz3DType::U64;
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
    version: Sz3CodecVersion,
}

/// Dtypes that SZ3 can compress and decompress
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
#[expect(missing_docs)]
pub enum Sz3DType {
    #[serde(rename = "u8", alias = "uint8")]
    U8,
    #[serde(rename = "u16", alias = "uint16")]
    U16,
    #[serde(rename = "u32", alias = "uint32")]
    U32,
    #[serde(rename = "u64", alias = "uint64")]
    U64,
    #[serde(rename = "i8", alias = "int8")]
    I8,
    #[serde(rename = "i16", alias = "int16")]
    I16,
    #[serde(rename = "i32", alias = "int32")]
    I32,
    #[serde(rename = "i64", alias = "int64")]
    I64,
    #[serde(rename = "f32", alias = "float32")]
    F32,
    #[serde(rename = "f64", alias = "float64")]
    F64,
}

impl Sz3DType {
    /// Get the corresponding [`AnyArrayDType`]
    #[must_use]
    pub const fn into_dtype(self) -> AnyArrayDType {
        match self {
            Self::U8 => AnyArrayDType::U8,
            Self::U16 => AnyArrayDType::U16,
            Self::U32 => AnyArrayDType::U32,
            Self::U64 => AnyArrayDType::U64,
            Self::I8 => AnyArrayDType::I8,
            Self::I16 => AnyArrayDType::I16,
            Self::I32 => AnyArrayDType::I32,
            Self::I64 => AnyArrayDType::I64,
            Self::F32 => AnyArrayDType::F32,
            Self::F64 => AnyArrayDType::F64,
        }
    }
}

impl fmt::Display for Sz3DType {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.write_str(match self {
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::I8 => "i8",
            Self::I16 => "i16",
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

        assert_eq!(decoded, AnyArray::I32(data.clone()));

        let mut decoded = Array::zeros(data.dim());
        decompress_into(&encoded, AnyArrayViewMut::I32(decoded.view_mut()))?;

        assert_eq!(decoded, data);

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

            let mut decoded = Array::zeros([data.len()]);
            decompress_into(
                &encoded,
                AnyArrayViewMut::F64(decoded.view_mut().into_dyn()),
            )?;

            assert_eq!(decoded, Array1::from_vec(data.to_vec()));
        }

        Ok(())
    }

    #[test]
    fn all_predictors() -> Result<(), Sz3CodecError> {
        let data = Array::linspace(-42.0, 42.0, 85);

        for predictor in [
            None,
            Some(Sz3Predictor::Interpolation),
            Some(Sz3Predictor::InterpolationLorenzo),
            Some(Sz3Predictor::Regression),
            Some(Sz3Predictor::LorenzoSecondOrder),
            Some(Sz3Predictor::LorenzoSecondOrderRegression),
            Some(Sz3Predictor::Lorenzo),
            Some(Sz3Predictor::LorenzoRegression),
            Some(Sz3Predictor::LorenzoFirstSecondOrder),
            Some(Sz3Predictor::LorenzoFirstSecondOrderRegression),
        ] {
            let encoded = compress(
                data.view(),
                predictor.as_ref(),
                &Sz3ErrorBound::Absolute { abs: 0.1 },
            )?;
            let _decoded = decompress(&encoded)?;

            let mut decoded = Array::zeros(data.dim());
            decompress_into(
                &encoded,
                AnyArrayViewMut::F64(decoded.view_mut().into_dyn()),
            )?;
        }

        Ok(())
    }

    #[test]
    fn all_dtypes() -> Result<(), Sz3CodecError> {
        fn compress_decompress<T: Sz3Element + num_traits::identities::Zero>(
            iter: impl Clone + IntoIterator<Item = T>,
            view_mut: impl for<'a> Fn(ArrayViewMut<'a, T, IxDyn>) -> AnyArrayViewMut<'a>,
        ) -> Result<(), Sz3CodecError> {
            let encoded = compress(
                Array::from_iter(iter.clone()).view(),
                default_predictor().as_ref(),
                &Sz3ErrorBound::Absolute { abs: 2.0 },
            )?;
            let _decoded = decompress(&encoded)?;

            let mut decoded = Array::<T, _>::zeros([iter.into_iter().count()]).into_dyn();
            decompress_into(&encoded, view_mut(decoded.view_mut().into_dyn()))?;

            Ok(())
        }

        compress_decompress(0_u8..42, |x| AnyArrayViewMut::U8(x))?;
        compress_decompress(-42_i8..42, |x| AnyArrayViewMut::I8(x))?;
        compress_decompress(0_u16..42, |x| AnyArrayViewMut::U16(x))?;
        compress_decompress(-42_i16..42, |x| AnyArrayViewMut::I16(x))?;
        compress_decompress(0_u32..42, |x| AnyArrayViewMut::U32(x))?;
        compress_decompress(-42_i32..42, |x| AnyArrayViewMut::I32(x))?;
        compress_decompress(0_u64..42, |x| AnyArrayViewMut::U64(x))?;
        compress_decompress(-42_i64..42, |x| AnyArrayViewMut::I64(x))?;
        compress_decompress((-42_i16..42).map(f32::from), |x| AnyArrayViewMut::F32(x))?;
        compress_decompress((-42_i16..42).map(f64::from), |x| AnyArrayViewMut::F64(x))?;

        Ok(())
    }
}
