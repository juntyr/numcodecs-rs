//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.85.0-blue
//! [repo]: https://github.com/juntyr/numcodecs-rs
//!
//! [Latest Version]: https://img.shields.io/crates/v/numcodecs-pco
//! [crates.io]: https://crates.io/crates/numcodecs-pco
//!
//! [Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs-pco
//! [docs.rs]: https://docs.rs/numcodecs-pco/
//!
//! [Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/numcodecs-rs/numcodecs_pco
//!
//! Pcodec implementation for the [`numcodecs`] API.

#![expect(clippy::multiple_crate_versions)] // embedded-io

use std::{borrow::Cow, fmt, num::NonZeroUsize};

use ndarray::{Array, Array1, ArrayBase, ArrayViewMut, Data, Dimension, ShapeError};
use numcodecs::{
    AnyArray, AnyArrayAssignError, AnyArrayDType, AnyArrayView, AnyArrayViewMut, AnyCowArray,
    Codec, StaticCodec, StaticCodecConfig, StaticCodecVersion,
};
use schemars::{JsonSchema, JsonSchema_repr};
use serde::{Deserialize, Serialize};
use serde_repr::{Deserialize_repr, Serialize_repr};
use thiserror::Error;

#[cfg(test)]
use ::serde_json as _;

type PcodecVersion = StaticCodecVersion<0, 1, 0>;

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(deny_unknown_fields)] // serde cannot deny unknown fields because of the flatten
/// Codec providing compression using pco
pub struct Pcodec {
    /// Compression level, ranging from 0 (weak) over 8 (very good) to 12
    /// (expensive)
    pub level: PcoCompressionLevel,
    /// Specifies how the mode should be determined
    #[serde(flatten)]
    pub mode: PcoModeSpec,
    /// Specifies how delta encoding should be chosen
    #[serde(flatten)]
    pub delta: PcoDeltaSpec,
    /// Specifies how the chunk should be split into pages
    #[serde(flatten)]
    pub paging: PcoPagingSpec,
    /// The codec's encoding format version. Do not provide this parameter explicitly.
    #[serde(default, rename = "_version")]
    pub version: PcodecVersion,
}

#[derive(
    Copy, Clone, Debug, Default, PartialEq, Eq, Serialize_repr, Deserialize_repr, JsonSchema_repr,
)]
#[repr(u8)]
/// Pco compression level.
///
/// The level ranges from 0 to 12 inclusive (default: 8):
/// * Level 0 achieves only a small amount of compression.
/// * Level 8 achieves very good compression.
/// * Level 12 achieves marginally better compression than 8.
#[expect(missing_docs)]
pub enum PcoCompressionLevel {
    Level0 = 0,
    Level1 = 1,
    Level2 = 2,
    Level3 = 3,
    Level4 = 4,
    Level5 = 5,
    Level6 = 6,
    Level7 = 7,
    #[default]
    Level8 = 8,
    Level9 = 9,
    Level10 = 10,
    Level11 = 11,
    Level12 = 12,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[schemars(deny_unknown_fields)] // serde cannot deny unknown fields because of the flatten
#[serde(tag = "mode", rename_all = "kebab-case")]
/// Pco compression mode
pub enum PcoModeSpec {
    #[default]
    /// Automatically detects a good mode.
    ///
    /// This works well most of the time, but costs some compression time and
    /// can select a bad mode in adversarial cases.
    Auto,
    /// Only uses the classic mode
    Classic,
    /// Tries using the `FloatMult` mode with a given base.
    ///
    /// Only applies to floating-point types.
    TryFloatMult {
        /// the base for the `FloatMult` mode
        float_mult_base: f64,
    },
    /// Tries using the `FloatQuant` mode with the given number of bits of
    /// quantization.
    ///
    /// Only applies to floating-point types.
    TryFloatQuant {
        /// the number of bits to which floating-point values are quantized
        float_quant_bits: u32,
    },
    /// Tries using the `IntMult` mode with a given base.
    ///
    /// Only applies to integer types.
    TryIntMult {
        /// the base for the `IntMult` mode
        int_mult_base: u64,
    },
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[schemars(deny_unknown_fields)] // serde cannot deny unknown fields because of the flatten
#[serde(tag = "delta", rename_all = "kebab-case")]
/// Pco delta encoding
pub enum PcoDeltaSpec {
    #[default]
    /// Automatically detects a detects a good delta encoding.
    ///
    /// This works well most of the time, but costs some compression time and
    /// can select a bad delta encoding in adversarial cases.
    Auto,
    /// Never uses delta encoding.
    ///
    /// This is best if your data is in a random order or adjacent numbers have
    /// no relation to each other.
    None,
    /// Tries taking nth order consecutive deltas.
    ///
    /// Supports a delta encoding order up to 7. For instance, 1st order is
    /// just regular delta encoding, 2nd is deltas-of-deltas, etc. It is legal
    /// to use 0th order, but it is identical to None.
    TryConsecutive {
        /// the order of the delta encoding
        delta_encoding_order: PcoDeltaEncodingOrder,
    },
    /// Tries delta encoding according to an extra latent variable of
    /// "lookback".
    ///
    /// This can improve compression ratio when there are nontrivial patterns
    /// in the array, but reduces compression speed substantially.
    TryLookback,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize_repr, Deserialize_repr, JsonSchema_repr)]
#[repr(u8)]
/// Pco delta encoding order.
///
/// The order ranges from 0 to 7 inclusive.
#[expect(missing_docs)]
pub enum PcoDeltaEncodingOrder {
    Order0 = 0,
    Order1 = 1,
    Order2 = 2,
    Order3 = 3,
    Order4 = 4,
    Order5 = 5,
    Order6 = 6,
    Order7 = 7,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[schemars(deny_unknown_fields)] // serde cannot deny unknown fields because of the flatten
#[serde(tag = "paging", rename_all = "kebab-case")]
/// Pco paging mode
pub enum PcoPagingSpec {
    /// Divide the chunk into equal pages of up to this many numbers.
    ///
    /// For example, with equal pages up to 100,000, a chunk of 150,000 numbers
    /// would be divided into 2 pages, each of 75,000 numbers.
    EqualPagesUpTo {
        #[serde(default = "default_equal_pages_up_to")]
        /// maximum amount of numbers in a page
        equal_pages_up_to: NonZeroUsize,
    },
}

impl Default for PcoPagingSpec {
    fn default() -> Self {
        Self::EqualPagesUpTo {
            equal_pages_up_to: default_equal_pages_up_to(),
        }
    }
}

const fn default_equal_pages_up_to() -> NonZeroUsize {
    NonZeroUsize::MIN.saturating_add(pco::DEFAULT_MAX_PAGE_N.saturating_sub(1))
}

impl Codec for Pcodec {
    type Error = PcodecError;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        match data {
            AnyCowArray::U16(data) => Ok(AnyArray::U8(
                Array1::from(compress(
                    data,
                    self.level,
                    self.mode,
                    self.delta,
                    self.paging,
                )?)
                .into_dyn(),
            )),
            AnyCowArray::U32(data) => Ok(AnyArray::U8(
                Array1::from(compress(
                    data,
                    self.level,
                    self.mode,
                    self.delta,
                    self.paging,
                )?)
                .into_dyn(),
            )),
            AnyCowArray::U64(data) => Ok(AnyArray::U8(
                Array1::from(compress(
                    data,
                    self.level,
                    self.mode,
                    self.delta,
                    self.paging,
                )?)
                .into_dyn(),
            )),
            AnyCowArray::I16(data) => Ok(AnyArray::U8(
                Array1::from(compress(
                    data,
                    self.level,
                    self.mode,
                    self.delta,
                    self.paging,
                )?)
                .into_dyn(),
            )),
            AnyCowArray::I32(data) => Ok(AnyArray::U8(
                Array1::from(compress(
                    data,
                    self.level,
                    self.mode,
                    self.delta,
                    self.paging,
                )?)
                .into_dyn(),
            )),
            AnyCowArray::I64(data) => Ok(AnyArray::U8(
                Array1::from(compress(
                    data,
                    self.level,
                    self.mode,
                    self.delta,
                    self.paging,
                )?)
                .into_dyn(),
            )),
            AnyCowArray::F32(data) => Ok(AnyArray::U8(
                Array1::from(compress(
                    data,
                    self.level,
                    self.mode,
                    self.delta,
                    self.paging,
                )?)
                .into_dyn(),
            )),
            AnyCowArray::F64(data) => Ok(AnyArray::U8(
                Array1::from(compress(
                    data,
                    self.level,
                    self.mode,
                    self.delta,
                    self.paging,
                )?)
                .into_dyn(),
            )),
            encoded => Err(PcodecError::UnsupportedDtype(encoded.dtype())),
        }
    }

    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
        let AnyCowArray::U8(encoded) = encoded else {
            return Err(PcodecError::EncodedDataNotBytes {
                dtype: encoded.dtype(),
            });
        };

        if !matches!(encoded.shape(), [_]) {
            return Err(PcodecError::EncodedDataNotOneDimensional {
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
            return Err(PcodecError::EncodedDataNotBytes {
                dtype: encoded.dtype(),
            });
        };

        if !matches!(encoded.shape(), [_]) {
            return Err(PcodecError::EncodedDataNotOneDimensional {
                shape: encoded.shape().to_vec(),
            });
        }

        let encoded = AnyArrayView::U8(encoded);
        let encoded = encoded.as_bytes();

        match decoded {
            AnyArrayViewMut::U16(decoded) => decompress_into(&encoded, decoded),
            AnyArrayViewMut::U32(decoded) => decompress_into(&encoded, decoded),
            AnyArrayViewMut::U64(decoded) => decompress_into(&encoded, decoded),
            AnyArrayViewMut::I16(decoded) => decompress_into(&encoded, decoded),
            AnyArrayViewMut::I32(decoded) => decompress_into(&encoded, decoded),
            AnyArrayViewMut::I64(decoded) => decompress_into(&encoded, decoded),
            AnyArrayViewMut::F32(decoded) => decompress_into(&encoded, decoded),
            AnyArrayViewMut::F64(decoded) => decompress_into(&encoded, decoded),
            decoded => Err(PcodecError::UnsupportedDtype(decoded.dtype())),
        }
    }
}

impl StaticCodec for Pcodec {
    const CODEC_ID: &'static str = "pco.rs";

    type Config<'de> = Self;

    fn from_config(config: Self::Config<'_>) -> Self {
        config
    }

    fn get_config(&self) -> StaticCodecConfig<Self> {
        StaticCodecConfig::from(self)
    }
}

#[derive(Debug, Error)]
/// Errors that may occur when applying the [`Pcodec`].
pub enum PcodecError {
    /// [`Pcodec`] does not support the dtype
    #[error("Pco does not support the dtype {0}")]
    UnsupportedDtype(AnyArrayDType),
    /// [`Pcodec`] failed to encode the header
    #[error("Pco failed to encode the header")]
    HeaderEncodeFailed {
        /// Opaque source error
        source: PcoHeaderError,
    },
    /// [`Pcodec`] failed to encode the data
    #[error("Pco failed to encode the data")]
    PcoEncodeFailed {
        /// Opaque source error
        source: PcoCodingError,
    },
    /// [`Pcodec`] can only decode one-dimensional byte arrays but received
    /// an array of a different dtype
    #[error(
        "Pco can only decode one-dimensional byte arrays but received an array of dtype {dtype}"
    )]
    EncodedDataNotBytes {
        /// The unexpected dtype of the encoded array
        dtype: AnyArrayDType,
    },
    /// [`Pcodec`] can only decode one-dimensional byte arrays but received
    /// an array of a different shape
    #[error(
        "Pco can only decode one-dimensional byte arrays but received a byte array of shape {shape:?}"
    )]
    EncodedDataNotOneDimensional {
        /// The unexpected shape of the encoded array
        shape: Vec<usize>,
    },
    /// [`Pcodec`] failed to decode the header
    #[error("Pco failed to decode the header")]
    HeaderDecodeFailed {
        /// Opaque source error
        source: PcoHeaderError,
    },
    /// [`Pcodec`] failed to decode the data
    #[error("Pco failed to decode the data")]
    PcoDecodeFailed {
        /// Opaque source error
        source: PcoCodingError,
    },
    /// [`Pcodec`] decoded an invalid array shape header which does not fit
    /// the decoded data
    #[error("Pco decoded an invalid array shape header which does not fit the decoded data")]
    DecodeInvalidShapeHeader {
        /// Source error
        #[from]
        source: ShapeError,
    },
    /// [`Pcodec`] cannot decode into the provided array
    #[error("Pco cannot decode into the provided array")]
    MismatchedDecodeIntoArray {
        /// The source of the error
        #[from]
        source: AnyArrayAssignError,
    },
}

#[derive(Debug, Error)]
#[error(transparent)]
/// Opaque error for when encoding or decoding the header fails
pub struct PcoHeaderError(postcard::Error);

#[derive(Debug, Error)]
#[error(transparent)]
/// Opaque error for when encoding or decoding with pco fails
pub struct PcoCodingError(pco::errors::PcoError);

#[expect(clippy::needless_pass_by_value)]
/// Compresses the input `data` array using pco with the given compression
/// `level`, `mode`, `delta` encoding, and `paging` mode.
///
/// # Errors
///
/// Errors with
/// - [`PcodecError::HeaderEncodeFailed`] if encoding the header failed
/// - [`PcodecError::PcoEncodeFailed`] if encoding failed with an opaque error
pub fn compress<T: PcoElement, S: Data<Elem = T>, D: Dimension>(
    data: ArrayBase<S, D>,
    level: PcoCompressionLevel,
    mode: PcoModeSpec,
    delta: PcoDeltaSpec,
    paging: PcoPagingSpec,
) -> Result<Vec<u8>, PcodecError> {
    let mut encoded_bytes = postcard::to_extend(
        &CompressionHeader {
            dtype: <T as PcoElement>::DTYPE,
            shape: Cow::Borrowed(data.shape()),
            version: StaticCodecVersion,
        },
        Vec::new(),
    )
    .map_err(|err| PcodecError::HeaderEncodeFailed {
        source: PcoHeaderError(err),
    })?;

    let data_owned;
    #[expect(clippy::option_if_let_else)]
    let data = if let Some(slice) = data.as_slice() {
        slice
    } else {
        data_owned = data.into_iter().copied().collect::<Vec<T>>();
        data_owned.as_slice()
    };

    let config = pco::ChunkConfig::default()
        .with_compression_level(level as usize)
        .with_mode_spec(match mode {
            PcoModeSpec::Auto => pco::ModeSpec::Auto,
            PcoModeSpec::Classic => pco::ModeSpec::Classic,
            PcoModeSpec::TryFloatMult { float_mult_base } => {
                pco::ModeSpec::TryFloatMult(float_mult_base)
            }
            PcoModeSpec::TryFloatQuant { float_quant_bits } => {
                pco::ModeSpec::TryFloatQuant(float_quant_bits)
            }
            PcoModeSpec::TryIntMult { int_mult_base } => pco::ModeSpec::TryIntMult(int_mult_base),
        })
        .with_delta_spec(match delta {
            PcoDeltaSpec::Auto => pco::DeltaSpec::Auto,
            PcoDeltaSpec::None => pco::DeltaSpec::None,
            PcoDeltaSpec::TryConsecutive {
                delta_encoding_order,
            } => pco::DeltaSpec::TryConsecutive(delta_encoding_order as usize),
            PcoDeltaSpec::TryLookback => pco::DeltaSpec::TryLookback,
        })
        .with_paging_spec(match paging {
            PcoPagingSpec::EqualPagesUpTo { equal_pages_up_to } => {
                pco::PagingSpec::EqualPagesUpTo(equal_pages_up_to.get())
            }
        });

    let encoded = pco::standalone::simple_compress(data, &config).map_err(|err| {
        PcodecError::PcoEncodeFailed {
            source: PcoCodingError(err),
        }
    })?;
    encoded_bytes.extend_from_slice(&encoded);

    Ok(encoded_bytes)
}

/// Decompresses the `encoded` data into an array.
///
/// # Errors
///
/// Errors with
/// - [`PcodecError::HeaderDecodeFailed`] if decoding the header failed
/// - [`PcodecError::PcoDecodeFailed`] if decoding failed with an opaque error
pub fn decompress(encoded: &[u8]) -> Result<AnyArray, PcodecError> {
    let (header, data) =
        postcard::take_from_bytes::<CompressionHeader>(encoded).map_err(|err| {
            PcodecError::HeaderDecodeFailed {
                source: PcoHeaderError(err),
            }
        })?;

    let decoded = match header.dtype {
        PcoDType::U16 => AnyArray::U16(Array::from_shape_vec(
            &*header.shape,
            pco::standalone::simple_decompress(data).map_err(|err| {
                PcodecError::PcoDecodeFailed {
                    source: PcoCodingError(err),
                }
            })?,
        )?),
        PcoDType::U32 => AnyArray::U32(Array::from_shape_vec(
            &*header.shape,
            pco::standalone::simple_decompress(data).map_err(|err| {
                PcodecError::PcoDecodeFailed {
                    source: PcoCodingError(err),
                }
            })?,
        )?),
        PcoDType::U64 => AnyArray::U64(Array::from_shape_vec(
            &*header.shape,
            pco::standalone::simple_decompress(data).map_err(|err| {
                PcodecError::PcoDecodeFailed {
                    source: PcoCodingError(err),
                }
            })?,
        )?),
        PcoDType::I16 => AnyArray::I16(Array::from_shape_vec(
            &*header.shape,
            pco::standalone::simple_decompress(data).map_err(|err| {
                PcodecError::PcoDecodeFailed {
                    source: PcoCodingError(err),
                }
            })?,
        )?),
        PcoDType::I32 => AnyArray::I32(Array::from_shape_vec(
            &*header.shape,
            pco::standalone::simple_decompress(data).map_err(|err| {
                PcodecError::PcoDecodeFailed {
                    source: PcoCodingError(err),
                }
            })?,
        )?),
        PcoDType::I64 => AnyArray::I64(Array::from_shape_vec(
            &*header.shape,
            pco::standalone::simple_decompress(data).map_err(|err| {
                PcodecError::PcoDecodeFailed {
                    source: PcoCodingError(err),
                }
            })?,
        )?),
        PcoDType::F32 => AnyArray::F32(Array::from_shape_vec(
            &*header.shape,
            pco::standalone::simple_decompress(data).map_err(|err| {
                PcodecError::PcoDecodeFailed {
                    source: PcoCodingError(err),
                }
            })?,
        )?),
        PcoDType::F64 => AnyArray::F64(Array::from_shape_vec(
            &*header.shape,
            pco::standalone::simple_decompress(data).map_err(|err| {
                PcodecError::PcoDecodeFailed {
                    source: PcoCodingError(err),
                }
            })?,
        )?),
    };

    Ok(decoded)
}

/// Decompresses the `encoded` data into the `decoded` array.
///
/// # Errors
///
/// Errors with
/// - [`PcodecError::HeaderDecodeFailed`] if decoding the header failed
/// - [`PcodecError::MismatchedDecodeIntoArray`] if the decoded array has the
///   wrong dtype or shape
/// - [`PcodecError::PcoDecodeFailed`] if decoding failed with an opaque error
/// - [`PcodecError::DecodeInvalidShapeHeader`] if the array shape header does
///   not fit the decoded data
pub fn decompress_into<T: PcoElement, D: Dimension>(
    encoded: &[u8],
    mut decoded: ArrayViewMut<T, D>,
) -> Result<(), PcodecError> {
    let (header, data) =
        postcard::take_from_bytes::<CompressionHeader>(encoded).map_err(|err| {
            PcodecError::HeaderDecodeFailed {
                source: PcoHeaderError(err),
            }
        })?;

    if T::DTYPE != header.dtype {
        return Err(PcodecError::MismatchedDecodeIntoArray {
            source: AnyArrayAssignError::DTypeMismatch {
                src: header.dtype.into_dtype(),
                dst: T::DTYPE.into_dtype(),
            },
        });
    }

    if decoded.shape() != &*header.shape {
        return Err(PcodecError::MismatchedDecodeIntoArray {
            source: AnyArrayAssignError::ShapeMismatch {
                src: header.shape.into_owned(),
                dst: decoded.shape().to_vec(),
            },
        });
    }

    if let Some(slice) = decoded.as_slice_mut() {
        pco::standalone::simple_decompress_into(data, slice).map_err(|err| {
            PcodecError::PcoDecodeFailed {
                source: PcoCodingError(err),
            }
        })?;
        return Ok(());
    }

    let dec =
        pco::standalone::simple_decompress(data).map_err(|err| PcodecError::PcoDecodeFailed {
            source: PcoCodingError(err),
        })?;

    if dec.len() != decoded.len() {
        return Err(PcodecError::DecodeInvalidShapeHeader {
            source: ShapeError::from_kind(ndarray::ErrorKind::IncompatibleShape),
        });
    }

    decoded.iter_mut().zip(dec).for_each(|(o, d)| *o = d);

    Ok(())
}

/// Array element types which can be compressed with pco.
pub trait PcoElement: Copy + pco::data_types::Number {
    /// The dtype representation of the type
    const DTYPE: PcoDType;
}

impl PcoElement for u16 {
    const DTYPE: PcoDType = PcoDType::U16;
}

impl PcoElement for u32 {
    const DTYPE: PcoDType = PcoDType::U32;
}

impl PcoElement for u64 {
    const DTYPE: PcoDType = PcoDType::U64;
}

impl PcoElement for i16 {
    const DTYPE: PcoDType = PcoDType::I16;
}

impl PcoElement for i32 {
    const DTYPE: PcoDType = PcoDType::I32;
}

impl PcoElement for i64 {
    const DTYPE: PcoDType = PcoDType::I64;
}

impl PcoElement for f32 {
    const DTYPE: PcoDType = PcoDType::F32;
}

impl PcoElement for f64 {
    const DTYPE: PcoDType = PcoDType::F64;
}

#[derive(Serialize, Deserialize)]
struct CompressionHeader<'a> {
    dtype: PcoDType,
    #[serde(borrow)]
    shape: Cow<'a, [usize]>,
    version: PcodecVersion,
}

/// Dtypes that pco can compress and decompress
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[expect(missing_docs)]
pub enum PcoDType {
    #[serde(rename = "u16", alias = "uint16")]
    U16,
    #[serde(rename = "u32", alias = "uint32")]
    U32,
    #[serde(rename = "u64", alias = "uint64")]
    U64,
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

impl PcoDType {
    #[must_use]
    /// Convert the [`PcoDType`] into an [`AnyArrayDType`]
    pub const fn into_dtype(self) -> AnyArrayDType {
        match self {
            Self::U16 => AnyArrayDType::U16,
            Self::U32 => AnyArrayDType::U32,
            Self::U64 => AnyArrayDType::U64,
            Self::I16 => AnyArrayDType::I16,
            Self::I32 => AnyArrayDType::I32,
            Self::I64 => AnyArrayDType::I64,
            Self::F32 => AnyArrayDType::F32,
            Self::F64 => AnyArrayDType::F64,
        }
    }
}

impl fmt::Display for PcoDType {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.write_str(match self {
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::F32 => "f32",
            Self::F64 => "f64",
        })
    }
}
