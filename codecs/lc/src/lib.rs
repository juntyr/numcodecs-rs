//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.87.0-blue
//! [repo]: https://github.com/juntyr/numcodecs-rs
//!
//! [Latest Version]: https://img.shields.io/crates/v/numcodecs-lc
//! [crates.io]: https://crates.io/crates/numcodecs-lc
//!
//! [Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs-lc
//! [docs.rs]: https://docs.rs/numcodecs-lc/
//!
//! [Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/numcodecs-rs/numcodecs_lc
//!
//! LC codec implementation for the [`numcodecs`] API.

#![allow(clippy::multiple_crate_versions)] // embedded-io

use std::borrow::Cow;

use ndarray::Array1;
use numcodecs::{
    AnyArray, AnyArrayAssignError, AnyArrayDType, AnyArrayView, AnyArrayViewMut, AnyCowArray,
    Codec, StaticCodec, StaticCodecConfig, StaticCodecVersion,
};
use schemars::{JsonSchema, JsonSchema_repr};
use serde::{Deserialize, Deserializer, Serialize};
use serde_repr::{Deserialize_repr, Serialize_repr};
use thiserror::Error;

#[cfg(test)]
use ::serde_json as _;

type LcCodecVersion = StaticCodecVersion<0, 1, 0>;

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
/// Codec providing compression using LC
pub struct LcCodec {
    /// LC preprocessors
    #[serde(default)]
    pub preprocessors: Vec<LcPreprocessor>,
    /// LC components
    #[serde(deserialize_with = "deserialize_components")]
    #[schemars(length(min = 1, max = lc_framework::MAX_COMPONENTS))]
    pub components: Vec<LcComponent>,
    /// The codec's encoding format version. Do not provide this parameter explicitly.
    #[serde(default, rename = "_version")]
    pub version: LcCodecVersion,
}

fn deserialize_components<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<LcComponent>, D::Error> {
    let components = Vec::<LcComponent>::deserialize(deserializer)?;

    if components.is_empty() {
        return Err(serde::de::Error::custom("expected at least one component"));
    }

    if components.len() > lc_framework::MAX_COMPONENTS {
        return Err(serde::de::Error::custom(format_args!(
            "expected at most {} components",
            lc_framework::MAX_COMPONENTS
        )));
    }

    Ok(components)
}

#[expect(missing_docs)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[serde(tag = "id")]
/// LC preprocessor
pub enum LcPreprocessor {
    #[serde(rename = "NUL")]
    Noop,
    #[serde(rename = "LOR")]
    Lorenzo1D { dtype: LcLorenzoDtype },
    #[serde(rename = "QUANT")]
    QuantizeErrorBound {
        dtype: LcQuantizeDType,
        kind: LcErrorKind,
        error_bound: f64,
        threshold: Option<f64>,
        decorrelation: LcDecorrelation,
    },
}

impl LcPreprocessor {
    const fn into_lc(self) -> lc_framework::Preprocessor {
        match self {
            Self::Noop => lc_framework::Preprocessor::Noop,
            Self::Lorenzo1D { dtype } => lc_framework::Preprocessor::Lorenzo1D {
                dtype: dtype.into_lc(),
            },
            Self::QuantizeErrorBound {
                dtype,
                kind,
                error_bound,
                threshold,
                decorrelation,
            } => lc_framework::Preprocessor::QuantizeErrorBound {
                dtype: dtype.into_lc(),
                kind: kind.into_lc(),
                error_bound,
                threshold,
                decorrelation: decorrelation.into_lc(),
            },
        }
    }
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
/// LC error bound kind
pub enum LcErrorKind {
    /// pointwise absolute error bound
    #[serde(rename = "ABS")]
    Abs,
    /// pointwise normalised absolute / data-range-relative error bound
    #[serde(rename = "NOA")]
    Noa,
    /// pointwise relative error bound
    #[serde(rename = "REL")]
    Rel,
}

impl LcErrorKind {
    const fn into_lc(self) -> lc_framework::ErrorKind {
        match self {
            Self::Abs => lc_framework::ErrorKind::Abs,
            Self::Noa => lc_framework::ErrorKind::Noa,
            Self::Rel => lc_framework::ErrorKind::Rel,
        }
    }
}

#[expect(missing_docs)]
#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
/// LC quantisation decorrelation mode
pub enum LcDecorrelation {
    #[serde(rename = "0")]
    Zero,
    #[serde(rename = "R")]
    Random,
}

impl LcDecorrelation {
    const fn into_lc(self) -> lc_framework::Decorrelation {
        match self {
            Self::Zero => lc_framework::Decorrelation::Zero,
            Self::Random => lc_framework::Decorrelation::Random,
        }
    }
}

#[expect(missing_docs)]
#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
/// LC Lorenzo preprocessor dtype
pub enum LcLorenzoDtype {
    #[serde(rename = "i32")]
    I32,
}

impl LcLorenzoDtype {
    const fn into_lc(self) -> lc_framework::LorenzoDtype {
        match self {
            Self::I32 => lc_framework::LorenzoDtype::I32,
        }
    }
}

#[expect(missing_docs)]
#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
/// LC quantization dtype
pub enum LcQuantizeDType {
    #[serde(rename = "f32")]
    F32,
    #[serde(rename = "f64")]
    F64,
}

impl LcQuantizeDType {
    const fn into_lc(self) -> lc_framework::QuantizeDType {
        match self {
            Self::F32 => lc_framework::QuantizeDType::F32,
            Self::F64 => lc_framework::QuantizeDType::F64,
        }
    }
}

#[expect(missing_docs)]
#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(deny_unknown_fields)]
#[serde(tag = "id")]
/// LC component
pub enum LcComponent {
    #[serde(rename = "NUL")]
    Noop,
    // mutators
    #[serde(rename = "TCMS")]
    TwosComplementToSignMagnitude { size: LcElemSize },
    #[serde(rename = "TCNB")]
    TwosComplementToNegaBinary { size: LcElemSize },
    #[serde(rename = "DBEFS")]
    DebiasedExponentFractionSign { size: LcFloatSize },
    #[serde(rename = "DBESF")]
    DebiasedExponentSignFraction { size: LcFloatSize },
    // shufflers
    #[serde(rename = "BIT")]
    BitShuffle { size: LcElemSize },
    #[serde(rename = "TUPL")]
    Tuple { size: LcTupleSize },
    // predictors
    #[serde(rename = "DIFF")]
    Delta { size: LcElemSize },
    #[serde(rename = "DIFFMS")]
    DeltaAsSignMagnitude { size: LcElemSize },
    #[serde(rename = "DIFFNB")]
    DeltaAsNegaBinary { size: LcElemSize },
    // reducers
    #[serde(rename = "CLOG")]
    Clog { size: LcElemSize },
    #[serde(rename = "HCLOG")]
    HClog { size: LcElemSize },
    #[serde(rename = "RARE")]
    Rare { size: LcElemSize },
    #[serde(rename = "RAZE")]
    Raze { size: LcElemSize },
    #[serde(rename = "RLE")]
    RunLengthEncoding { size: LcElemSize },
    #[serde(rename = "RRE")]
    RepetitionRunBitmapEncoding { size: LcElemSize },
    #[serde(rename = "RZE")]
    ZeroRunBitmapEncoding { size: LcElemSize },
}

impl LcComponent {
    const fn into_lc(self) -> lc_framework::Component {
        match self {
            Self::Noop => lc_framework::Component::Noop,
            // mutators
            Self::TwosComplementToSignMagnitude { size } => {
                lc_framework::Component::TwosComplementToSignMagnitude {
                    size: size.into_lc(),
                }
            }
            Self::TwosComplementToNegaBinary { size } => {
                lc_framework::Component::TwosComplementToNegaBinary {
                    size: size.into_lc(),
                }
            }
            Self::DebiasedExponentFractionSign { size } => {
                lc_framework::Component::DebiasedExponentFractionSign {
                    size: size.into_lc(),
                }
            }
            Self::DebiasedExponentSignFraction { size } => {
                lc_framework::Component::DebiasedExponentSignFraction {
                    size: size.into_lc(),
                }
            }
            // shufflers
            Self::BitShuffle { size } => lc_framework::Component::BitShuffle {
                size: size.into_lc(),
            },
            Self::Tuple { size } => lc_framework::Component::Tuple {
                size: size.into_lc(),
            },
            // predictors
            Self::Delta { size } => lc_framework::Component::Delta {
                size: size.into_lc(),
            },
            Self::DeltaAsSignMagnitude { size } => lc_framework::Component::DeltaAsSignMagnitude {
                size: size.into_lc(),
            },
            Self::DeltaAsNegaBinary { size } => lc_framework::Component::DeltaAsNegaBinary {
                size: size.into_lc(),
            },
            // reducers
            Self::Clog { size } => lc_framework::Component::Clog {
                size: size.into_lc(),
            },
            Self::HClog { size } => lc_framework::Component::HClog {
                size: size.into_lc(),
            },
            Self::Rare { size } => lc_framework::Component::Rare {
                size: size.into_lc(),
            },
            Self::Raze { size } => lc_framework::Component::Raze {
                size: size.into_lc(),
            },
            Self::RunLengthEncoding { size } => lc_framework::Component::RunLengthEncoding {
                size: size.into_lc(),
            },
            Self::RepetitionRunBitmapEncoding { size } => {
                lc_framework::Component::RepetitionRunBitmapEncoding {
                    size: size.into_lc(),
                }
            }
            Self::ZeroRunBitmapEncoding { size } => {
                lc_framework::Component::ZeroRunBitmapEncoding {
                    size: size.into_lc(),
                }
            }
        }
    }
}

#[expect(missing_docs)]
#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize_repr,
    Deserialize_repr,
    JsonSchema_repr,
)]
/// LC component element size, in bytes
#[repr(u8)]
pub enum LcElemSize {
    S1 = 1,
    S2 = 2,
    S4 = 4,
    S8 = 8,
}

impl LcElemSize {
    const fn into_lc(self) -> lc_framework::ElemSize {
        match self {
            Self::S1 => lc_framework::ElemSize::S1,
            Self::S2 => lc_framework::ElemSize::S2,
            Self::S4 => lc_framework::ElemSize::S4,
            Self::S8 => lc_framework::ElemSize::S8,
        }
    }
}

#[expect(missing_docs)]
#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize_repr,
    Deserialize_repr,
    JsonSchema_repr,
)]
/// LC component float element size, in bytes
#[repr(u8)]
pub enum LcFloatSize {
    S4 = 4,
    S8 = 8,
}

impl LcFloatSize {
    const fn into_lc(self) -> lc_framework::FloatSize {
        match self {
            Self::S4 => lc_framework::FloatSize::S4,
            Self::S8 => lc_framework::FloatSize::S8,
        }
    }
}

#[expect(missing_docs)]
#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
/// LC tuple component element size, in bytes x tuple length
#[schemars(description = "LC tuple component element size, in tuple length _ bytes")]
pub enum LcTupleSize {
    #[serde(rename = "2_1")]
    S1x2,
    #[serde(rename = "3_1")]
    S1x3,
    #[serde(rename = "4_1")]
    S1x4,
    #[serde(rename = "6_1")]
    S1x6,
    #[serde(rename = "8_1")]
    S1x8,
    #[serde(rename = "12_1")]
    S1x12,
    #[serde(rename = "2_2")]
    S2x2,
    #[serde(rename = "3_2")]
    S2x3,
    #[serde(rename = "4_2")]
    S2x4,
    #[serde(rename = "6_2")]
    S2x6,
    #[serde(rename = "2_4")]
    S4x2,
    #[serde(rename = "6_4")]
    S4x6,
    #[serde(rename = "3_8")]
    S8x3,
    #[serde(rename = "6_8")]
    S8x6,
}

impl LcTupleSize {
    const fn into_lc(self) -> lc_framework::TupleSize {
        match self {
            Self::S1x2 => lc_framework::TupleSize::S1x2,
            Self::S1x3 => lc_framework::TupleSize::S1x3,
            Self::S1x4 => lc_framework::TupleSize::S1x4,
            Self::S1x6 => lc_framework::TupleSize::S1x6,
            Self::S1x8 => lc_framework::TupleSize::S1x8,
            Self::S1x12 => lc_framework::TupleSize::S1x12,
            Self::S2x2 => lc_framework::TupleSize::S2x2,
            Self::S2x3 => lc_framework::TupleSize::S2x3,
            Self::S2x4 => lc_framework::TupleSize::S2x4,
            Self::S2x6 => lc_framework::TupleSize::S2x6,
            Self::S4x2 => lc_framework::TupleSize::S4x2,
            Self::S4x6 => lc_framework::TupleSize::S4x6,
            Self::S8x3 => lc_framework::TupleSize::S8x3,
            Self::S8x6 => lc_framework::TupleSize::S8x6,
        }
    }
}

impl Codec for LcCodec {
    type Error = LcCodecError;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        compress(data.view(), &self.preprocessors, &self.components)
            .map(|bytes| AnyArray::U8(Array1::from_vec(bytes).into_dyn()))
    }

    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
        let AnyCowArray::U8(encoded) = encoded else {
            return Err(LcCodecError::EncodedDataNotBytes {
                dtype: encoded.dtype(),
            });
        };

        if !matches!(encoded.shape(), [_]) {
            return Err(LcCodecError::EncodedDataNotOneDimensional {
                shape: encoded.shape().to_vec(),
            });
        }

        decompress(
            &self.preprocessors,
            &self.components,
            &AnyCowArray::U8(encoded).as_bytes(),
        )
    }

    fn decode_into(
        &self,
        encoded: AnyArrayView,
        decoded: AnyArrayViewMut,
    ) -> Result<(), Self::Error> {
        let AnyArrayView::U8(encoded) = encoded else {
            return Err(LcCodecError::EncodedDataNotBytes {
                dtype: encoded.dtype(),
            });
        };

        if !matches!(encoded.shape(), [_]) {
            return Err(LcCodecError::EncodedDataNotOneDimensional {
                shape: encoded.shape().to_vec(),
            });
        }

        decompress_into(
            &self.preprocessors,
            &self.components,
            &AnyArrayView::U8(encoded).as_bytes(),
            decoded,
        )
    }
}

impl StaticCodec for LcCodec {
    const CODEC_ID: &'static str = "lc.rs";

    type Config<'de> = Self;

    fn from_config(config: Self::Config<'_>) -> Self {
        config
    }

    fn get_config(&self) -> StaticCodecConfig<'_, Self> {
        StaticCodecConfig::from(self)
    }
}

#[derive(Debug, Error)]
/// Errors that may occur when applying the [`LcCodec`].
pub enum LcCodecError {
    /// [`LcCodec`] failed to encode the header
    #[error("Lc failed to encode the header")]
    HeaderEncodeFailed {
        /// Opaque source error
        source: LcHeaderError,
    },
    /// [`LcCodec`] failed to encode the encoded data
    #[error("Lc failed to decode the encoded data")]
    LcEncodeFailed {
        /// Opaque source error
        source: LcCodingError,
    },
    /// [`LcCodec`] can only decode one-dimensional byte arrays but received
    /// an array of a different dtype
    #[error(
        "Lc can only decode one-dimensional byte arrays but received an array of dtype {dtype}"
    )]
    EncodedDataNotBytes {
        /// The unexpected dtype of the encoded array
        dtype: AnyArrayDType,
    },
    /// [`LcCodec`] can only decode one-dimensional byte arrays but received
    /// an array of a different shape
    #[error(
        "Lc can only decode one-dimensional byte arrays but received a byte array of shape {shape:?}"
    )]
    EncodedDataNotOneDimensional {
        /// The unexpected shape of the encoded array
        shape: Vec<usize>,
    },
    /// [`LcCodec`] failed to encode the header
    #[error("Lc failed to decode the header")]
    HeaderDecodeFailed {
        /// Opaque source error
        source: LcHeaderError,
    },
    /// [`LcCodec`] decode produced a different number of bytes than expected
    #[error("Lc decode produced a different number of bytes than expected")]
    DecodeDataLengthMismatch,
    /// [`LcCodec`] failed to decode the encoded data
    #[error("Lc failed to decode the encoded data")]
    LcDecodeFailed {
        /// Opaque source error
        source: LcCodingError,
    },
    /// [`LcCodec`] cannot decode into the provided array
    #[error("Lc cannot decode into the provided array")]
    MismatchedDecodeIntoArray {
        /// The source of the error
        #[from]
        source: AnyArrayAssignError,
    },
}

#[derive(Debug, Error)]
#[error(transparent)]
/// Opaque error for when encoding or decoding the header fails
pub struct LcHeaderError(postcard::Error);

#[derive(Debug, Error)]
#[error(transparent)]
/// Opaque error for when encoding or decoding with LC fails
pub struct LcCodingError(lc_framework::Error);

#[expect(clippy::needless_pass_by_value)]
/// Compress the `array` using LC with the provided `preprocessors` and
/// `components`.
///
/// # Errors
///
/// Errors with
/// - [`LcCodecError::HeaderEncodeFailed`] if encoding the header to the
///   output bytevec failed
/// - [`LcCodecError::LcEncodeFailed`] if an opaque encoding error occurred
pub fn compress(
    array: AnyArrayView,
    preprocessors: &[LcPreprocessor],
    components: &[LcComponent],
) -> Result<Vec<u8>, LcCodecError> {
    let mut encoded = postcard::to_extend(
        &CompressionHeader {
            dtype: array.dtype(),
            shape: Cow::Borrowed(array.shape()),
            version: StaticCodecVersion,
        },
        Vec::new(),
    )
    .map_err(|err| LcCodecError::HeaderEncodeFailed {
        source: LcHeaderError(err),
    })?;

    // LC does not support empty input, so skip encoding
    if array.is_empty() {
        return Ok(encoded);
    }

    let preprocessors = preprocessors
        .iter()
        .cloned()
        .map(LcPreprocessor::into_lc)
        .collect::<Vec<_>>();
    let components = components
        .iter()
        .copied()
        .map(LcComponent::into_lc)
        .collect::<Vec<_>>();

    encoded.append(
        &mut lc_framework::compress(&preprocessors, &components, &array.as_bytes()).map_err(
            |err| LcCodecError::LcEncodeFailed {
                source: LcCodingError(err),
            },
        )?,
    );

    Ok(encoded)
}

/// Decompress the `encoded` data into an array using LC.
///
/// # Errors
///
/// Errors with
/// - [`LcCodecError::HeaderDecodeFailed`] if decoding the header failed
/// - [`LcCodecError::DecodeDataLengthMismatch`] if decoding produced a
///   different number of bytes than expected
/// - [`LcCodecError::LcDecodeFailed`] if an opaque decoding error occurred
pub fn decompress(
    preprocessors: &[LcPreprocessor],
    components: &[LcComponent],
    encoded: &[u8],
) -> Result<AnyArray, LcCodecError> {
    let (header, encoded) =
        postcard::take_from_bytes::<CompressionHeader>(encoded).map_err(|err| {
            LcCodecError::HeaderDecodeFailed {
                source: LcHeaderError(err),
            }
        })?;

    let (decoded, result) = AnyArray::with_zeros_bytes(header.dtype, &header.shape, |decoded| {
        decompress_into_bytes(preprocessors, components, encoded, decoded)
    });

    result.map(|()| decoded)
}

/// Decompress the `encoded` data into a `decoded` array using LC.
///
/// # Errors
///
/// Errors with
/// - [`LcCodecError::HeaderDecodeFailed`] if decoding the header failed
/// - [`LcCodecError::MismatchedDecodeIntoArray`] if the `decoded` array is of
///   the wrong dtype or shape
/// - [`LcCodecError::HeaderDecodeFailed`] if decoding the header failed
/// - [`LcCodecError::DecodeDataLengthMismatch`] if decoding produced a
///   different number of bytes than expected
/// - [`LcCodecError::LcDecodeFailed`] if an opaque decoding error occurred
pub fn decompress_into(
    preprocessors: &[LcPreprocessor],
    components: &[LcComponent],
    encoded: &[u8],
    mut decoded: AnyArrayViewMut,
) -> Result<(), LcCodecError> {
    let (header, encoded) =
        postcard::take_from_bytes::<CompressionHeader>(encoded).map_err(|err| {
            LcCodecError::HeaderDecodeFailed {
                source: LcHeaderError(err),
            }
        })?;

    if header.dtype != decoded.dtype() {
        return Err(LcCodecError::MismatchedDecodeIntoArray {
            source: AnyArrayAssignError::DTypeMismatch {
                src: header.dtype,
                dst: decoded.dtype(),
            },
        });
    }

    if header.shape != decoded.shape() {
        return Err(LcCodecError::MismatchedDecodeIntoArray {
            source: AnyArrayAssignError::ShapeMismatch {
                src: header.shape.into_owned(),
                dst: decoded.shape().to_vec(),
            },
        });
    }

    decoded.with_bytes_mut(|decoded| {
        decompress_into_bytes(preprocessors, components, encoded, decoded)
    })
}

fn decompress_into_bytes(
    preprocessors: &[LcPreprocessor],
    components: &[LcComponent],
    encoded: &[u8],
    decoded: &mut [u8],
) -> Result<(), LcCodecError> {
    // LC does not support empty input, so skip decoding
    if decoded.is_empty() && encoded.is_empty() {
        return Ok(());
    }

    let preprocessors = preprocessors
        .iter()
        .cloned()
        .map(LcPreprocessor::into_lc)
        .collect::<Vec<_>>();
    let components = components
        .iter()
        .copied()
        .map(LcComponent::into_lc)
        .collect::<Vec<_>>();

    let dec = lc_framework::decompress(&preprocessors, &components, encoded).map_err(|err| {
        LcCodecError::LcDecodeFailed {
            source: LcCodingError(err),
        }
    })?;

    if dec.len() != decoded.len() {
        return Err(LcCodecError::DecodeDataLengthMismatch);
    }

    decoded.copy_from_slice(&dec);

    Ok(())
}

#[derive(Serialize, Deserialize)]
struct CompressionHeader<'a> {
    dtype: AnyArrayDType,
    #[serde(borrow)]
    shape: Cow<'a, [usize]>,
    version: LcCodecVersion,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn lossless() {
        let data = ndarray::linspace(0.0, std::f32::consts::PI, 100)
            .collect::<Array1<f32>>()
            .into_shape_with_order((10, 10))
            .unwrap()
            .cos();

        let preprocessors = &[];
        let components = &[
            LcComponent::BitShuffle {
                size: LcElemSize::S4,
            },
            LcComponent::RunLengthEncoding {
                size: LcElemSize::S4,
            },
        ];

        let compressed = compress(
            AnyArrayView::F32(data.view().into_dyn()),
            preprocessors,
            components,
        )
        .unwrap();
        let decompressed = decompress(preprocessors, components, &compressed).unwrap();

        assert_eq!(decompressed, AnyArray::F32(data.into_dyn()));
    }

    #[test]
    fn abs_error() {
        let data = ndarray::linspace(0.0, std::f32::consts::PI, 100)
            .collect::<Array1<f32>>()
            .into_shape_with_order((10, 10))
            .unwrap()
            .cos();

        let preprocessors = &[LcPreprocessor::QuantizeErrorBound {
            dtype: LcQuantizeDType::F32,
            kind: LcErrorKind::Abs,
            error_bound: 0.1,
            threshold: None,
            decorrelation: LcDecorrelation::Zero,
        }];
        let components = &[
            LcComponent::BitShuffle {
                size: LcElemSize::S4,
            },
            LcComponent::RunLengthEncoding {
                size: LcElemSize::S4,
            },
        ];

        let compressed = compress(
            AnyArrayView::F32(data.view().into_dyn()),
            preprocessors,
            components,
        )
        .unwrap();
        let decompressed = decompress(preprocessors, components, &compressed).unwrap();

        let AnyArray::F32(decompressed) = decompressed else {
            panic!("unexpected decompressed dtype {}", decompressed.dtype());
        };
        assert_eq!(decompressed.shape(), data.shape());

        for (o, d) in data.into_iter().zip(decompressed) {
            assert!((o - d).abs() <= 0.1);
        }
    }
}
