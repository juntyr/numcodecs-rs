//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.82.0-blue
//! [repo]: https://github.com/juntyr/numcodecs-rs
//!
//! [Latest Version]: https://img.shields.io/crates/v/numcodecs-zfp
//! [crates.io]: https://crates.io/crates/numcodecs-zfp
//!
//! [Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs-zfp
//! [docs.rs]: https://docs.rs/numcodecs-zfp/
//!
//! [Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/numcodecs-rs/numcodecs_zfp
//!
//! ZFP codec implementation for the [`numcodecs`] API.

use std::{borrow::Cow, fmt};

use ndarray::{Array, Array1, ArrayView, Dimension, Zip};
use numcodecs::{
    AnyArray, AnyArrayAssignError, AnyArrayDType, AnyArrayView, AnyArrayViewMut, AnyCowArray,
    Codec, StaticCodec, StaticCodecConfig,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;

mod ffi;

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
#[serde(transparent)]
/// Codec providing compression using ZFP
pub struct ZfpCodec {
    /// ZFP compression mode
    pub mode: ZfpCompressionMode,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "mode")]
#[serde(deny_unknown_fields)]
/// ZFP compression mode
pub enum ZfpCompressionMode {
    #[serde(rename = "expert")]
    /// The most general mode, which can describe all four other modes
    Expert {
        /// Minimum number of compressed bits used to represent a block
        min_bits: u32,
        /// Maximum number of bits used to represent a block
        max_bits: u32,
        /// Maximum number of bit planes encoded
        max_prec: u32,
        /// Smallest absolute bit plane number encoded.
        ///
        /// This parameter applies to floating-point data only and is ignored
        /// for integer data.
        min_exp: i32,
    },
    /// In fixed-rate mode, each d-dimensional compressed block of `$4^d$`
    /// values is stored using a fixed number of bits. This number of
    /// compressed bits per block is amortized over the `$4^d$` values to give
    /// a rate of `$rate = \frac{maxbits}{4^d}$` in bits per value.
    #[serde(rename = "fixed-rate")]
    FixedRate {
        /// Rate in bits per value
        rate: f64,
    },
    /// In fixed-precision mode, the number of bits used to encode a block may
    /// vary, but the number of bit planes (the precision) encoded for the
    /// transform coefficients is fixed.
    #[serde(rename = "fixed-precision")]
    FixedPrecision {
        /// Number of bit planes encoded
        precision: u32,
    },
    /// In fixed-accuracy mode, all transform coefficient bit planes up to a
    /// minimum bit plane number are encoded. The smallest absolute bit plane
    /// number is chosen such that
    /// `$minexp = \text{floor}(\log_{2}(tolerance))$`.
    #[serde(rename = "fixed-accuracy")]
    FixedAccuracy {
        /// Absolute error tolerance
        tolerance: f64,
    },
    /// Lossless per-block compression that preserves integer and floating point
    /// bit patterns.
    #[serde(rename = "reversible")]
    Reversible,
}

impl Codec for ZfpCodec {
    type Error = ZfpCodecError;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        if matches!(data.dtype(), AnyArrayDType::I32 | AnyArrayDType::I64)
            && matches!(
                self.mode,
                ZfpCompressionMode::FixedAccuracy { tolerance: _ }
            )
        {
            return Err(ZfpCodecError::FixedAccuracyModeIntegerData);
        }

        match data {
            AnyCowArray::I32(data) => Ok(AnyArray::U8(
                Array1::from(compress(data.view(), &self.mode)?).into_dyn(),
            )),
            AnyCowArray::I64(data) => Ok(AnyArray::U8(
                Array1::from(compress(data.view(), &self.mode)?).into_dyn(),
            )),
            AnyCowArray::F32(data) => Ok(AnyArray::U8(
                Array1::from(compress(data.view(), &self.mode)?).into_dyn(),
            )),
            AnyCowArray::F64(data) => Ok(AnyArray::U8(
                Array1::from(compress(data.view(), &self.mode)?).into_dyn(),
            )),
            encoded => Err(ZfpCodecError::UnsupportedDtype(encoded.dtype())),
        }
    }

    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
        let AnyCowArray::U8(encoded) = encoded else {
            return Err(ZfpCodecError::EncodedDataNotBytes {
                dtype: encoded.dtype(),
            });
        };

        if !matches!(encoded.shape(), [_]) {
            return Err(ZfpCodecError::EncodedDataNotOneDimensional {
                shape: encoded.shape().to_vec(),
            });
        }

        decompress(&AnyCowArray::U8(encoded).as_bytes(), &self.mode)
    }

    fn decode_into(
        &self,
        encoded: AnyArrayView,
        decoded: AnyArrayViewMut,
    ) -> Result<(), Self::Error> {
        let AnyArrayView::U8(encoded) = encoded else {
            return Err(ZfpCodecError::EncodedDataNotBytes {
                dtype: encoded.dtype(),
            });
        };

        if !matches!(encoded.shape(), [_]) {
            return Err(ZfpCodecError::EncodedDataNotOneDimensional {
                shape: encoded.shape().to_vec(),
            });
        }

        decompress_into(&AnyArrayView::U8(encoded).as_bytes(), &self.mode, decoded)
    }
}

impl StaticCodec for ZfpCodec {
    const CODEC_ID: &'static str = "zfp";

    type Config<'de> = Self;

    fn from_config(config: Self::Config<'_>) -> Self {
        config
    }

    fn get_config(&self) -> StaticCodecConfig<Self> {
        StaticCodecConfig::from(self)
    }
}

#[derive(Debug, Error)]
/// Errors that may occur when applying the [`ZfpCodec`].
pub enum ZfpCodecError {
    /// [`ZfpCodec`] does not support the dtype
    #[error("Zfp does not support the dtype {0}")]
    UnsupportedDtype(AnyArrayDType),
    /// [`ZfpCodec`] does not support the fixed accuracy mode for integer data
    #[error("Zfp does not support the fixed accuracy mode for integer data")]
    FixedAccuracyModeIntegerData,
    /// [`ZfpCodec`] only supports 1-4 dimensional data
    #[error("Zfp only supports 1-4 dimensional data but found shape {shape:?}")]
    ExcessiveDimensionality {
        /// The unexpected shape of the data
        shape: Vec<usize>,
    },
    /// [`ZfpCodec`] was configured with an invalid expert `mode`
    #[error("Zfp was configured with an invalid expert mode {mode:?}")]
    InvalidExpertMode {
        /// The unexpected compression mode
        mode: ZfpCompressionMode,
    },
    /// [`ZfpCodec`] does not support non-finite (infinite or NaN) floating
    /// point data  in non-reversible lossy compression
    #[error("Zfp does not support non-finite (infinite or NaN) floating point data in non-reversible lossy compression")]
    NonFiniteData,
    /// [`ZfpCodec`] failed to encode the header
    #[error("Zfp failed to encode the header")]
    HeaderEncodeFailed {
        /// Opaque source error
        source: ZfpHeaderError,
    },
    /// [`ZfpCodec`] failed to encode the data
    #[error("Zfp failed to encode the data")]
    ZfpEncodeFailed,
    /// [`ZfpCodec`] can only decode one-dimensional byte arrays but received
    /// an array of a different dtype
    #[error(
        "Zfp can only decode one-dimensional byte arrays but received an array of dtype {dtype}"
    )]
    EncodedDataNotBytes {
        /// The unexpected dtype of the encoded array
        dtype: AnyArrayDType,
    },
    /// [`ZfpCodec`] can only decode one-dimensional byte arrays but received
    /// an array of a different shape
    #[error("Zfp can only decode one-dimensional byte arrays but received a byte array of shape {shape:?}")]
    EncodedDataNotOneDimensional {
        /// The unexpected shape of the encoded array
        shape: Vec<usize>,
    },
    /// [`ZfpCodec`] failed to decode the header
    #[error("Zfp failed to decode the header")]
    HeaderDecodeFailed {
        /// Opaque source error
        source: ZfpHeaderError,
    },
    /// [`ZfpCodec`] cannot decode into the provided array
    #[error("ZfpCodec cannot decode into the provided array")]
    MismatchedDecodeIntoArray {
        /// The source of the error
        #[from]
        source: AnyArrayAssignError,
    },
    /// [`ZfpCodec`] failed to decode the data
    #[error("Zfp failed to decode the data")]
    ZfpDecodeFailed,
}

#[derive(Debug, Error)]
#[error(transparent)]
/// Opaque error for when encoding or decoding the header fails
pub struct ZfpHeaderError(postcard::Error);

/// Compress the `data` array using ZFP with the provided `mode`.
///
/// # Errors
///
/// Errors with
/// - [`ZfpCodecError::NonFiniteData`] if any data element is non-finite
///   (infinite or NaN) and a non-reversible lossy compression `mode` is used
/// - [`ZfpCodecError::ExcessiveDimensionality`] if data is more than
///   4-dimensional
/// - [`ZfpCodecError::InvalidExpertMode`] if the `mode` has invalid expert mode
///   parameters
/// - [`ZfpCodecError::HeaderEncodeFailed`] if encoding the ZFP header failed
/// - [`ZfpCodecError::ZfpEncodeFailed`] if an opaque encoding error occurred
pub fn compress<T: ffi::ZfpCompressible, D: Dimension>(
    data: ArrayView<T, D>,
    mode: &ZfpCompressionMode,
) -> Result<Vec<u8>, ZfpCodecError> {
    if !matches!(mode, ZfpCompressionMode::Reversible) && !Zip::from(&data).all(|x| x.is_finite()) {
        return Err(ZfpCodecError::NonFiniteData);
    }

    let mut encoded = postcard::to_extend(
        &CompressionHeader {
            dtype: <T as ffi::ZfpCompressible>::D_TYPE,
            shape: Cow::Borrowed(data.shape()),
        },
        Vec::new(),
    )
    .map_err(|err| ZfpCodecError::HeaderEncodeFailed {
        source: ZfpHeaderError(err),
    })?;

    // Setup zfp structs to begin compression
    // Squeeze the data to avoid wasting ZFP dimensions on axes of length 1
    let field = ffi::ZfpField::new(data.into_dyn().squeeze())?;
    let stream = ffi::ZfpCompressionStream::new(&field, mode)?;

    // Allocate space based on the maximum size potentially required by zfp to
    //  store the compressed array
    let stream = stream.with_bitstream(field, &mut encoded);

    // Compress the field into the allocated output array
    stream.compress()?;

    Ok(encoded)
}

/// Decompress the `encoded` data into an array using ZFP with the provided `mode`.
///
/// # Errors
///
/// Errors with
/// - [`ZfpCodecError::HeaderDecodeFailed`] if decoding the header failed
/// - [`ZfpCodecError::DecodeUnknownDtype`] if the encoded data uses an unknown
///   dtype
/// - [`ZfpCodecError::ZfpDecodeFailed`] if an opaque decoding error occurred
pub fn decompress(encoded: &[u8], mode: &ZfpCompressionMode) -> Result<AnyArray, ZfpCodecError> {
    let (header, encoded) =
        postcard::take_from_bytes::<CompressionHeader>(encoded).map_err(|err| {
            ZfpCodecError::HeaderDecodeFailed {
                source: ZfpHeaderError(err),
            }
        })?;

    // Setup zfp structs to begin decompression
    let stream = ffi::ZfpDecompressionStream::new(encoded);

    // Decompress the field into a newly allocated output array
    match header.dtype {
        ZfpDType::I32 => {
            let mut decompressed = Array::zeros(&*header.shape);
            stream.decompress(mode, decompressed.view_mut().squeeze())?;
            Ok(AnyArray::I32(decompressed))
        }
        ZfpDType::I64 => {
            let mut decompressed = Array::zeros(&*header.shape);
            stream.decompress(mode, decompressed.view_mut().squeeze())?;
            Ok(AnyArray::I64(decompressed))
        }
        ZfpDType::F32 => {
            let mut decompressed = Array::zeros(&*header.shape);
            stream.decompress(mode, decompressed.view_mut().squeeze())?;
            Ok(AnyArray::F32(decompressed))
        }
        ZfpDType::F64 => {
            let mut decompressed = Array::zeros(&*header.shape);
            stream.decompress(mode, decompressed.view_mut().squeeze())?;
            Ok(AnyArray::F64(decompressed))
        }
    }
}

/// Decompress the `encoded` data into a `decoded` array using ZFP with the
/// provided `mode`.
///
/// # Errors
///
/// Errors with
/// - [`ZfpCodecError::HeaderDecodeFailed`] if decoding the header failed
/// - [`ZfpCodecError::DecodeUnknownDtype`] if the encoded data uses an unknown
///   dtype
/// - [`ZfpCodecError::MismatchedDecodeIntoArray`] if the `decoded` array is of
///   the wrong dtype or shape
/// - [`ZfpCodecError::ZfpDecodeFailed`] if an opaque decoding error occurred
pub fn decompress_into(
    encoded: &[u8],
    mode: &ZfpCompressionMode,
    decoded: AnyArrayViewMut,
) -> Result<(), ZfpCodecError> {
    let (header, encoded) =
        postcard::take_from_bytes::<CompressionHeader>(encoded).map_err(|err| {
            ZfpCodecError::HeaderDecodeFailed {
                source: ZfpHeaderError(err),
            }
        })?;

    if decoded.shape() != &*header.shape {
        return Err(ZfpCodecError::MismatchedDecodeIntoArray {
            source: AnyArrayAssignError::ShapeMismatch {
                src: header.shape.into_owned(),
                dst: decoded.shape().to_vec(),
            },
        });
    }

    // Setup zfp structs to begin decompression
    let stream = ffi::ZfpDecompressionStream::new(encoded);

    // Decompress the field into the output array
    match (decoded, header.dtype) {
        (AnyArrayViewMut::I32(decoded), ZfpDType::I32) => {
            stream.decompress(mode, decoded.squeeze())
        }
        (AnyArrayViewMut::I64(decoded), ZfpDType::I64) => {
            stream.decompress(mode, decoded.squeeze())
        }
        (AnyArrayViewMut::F32(decoded), ZfpDType::F32) => {
            stream.decompress(mode, decoded.squeeze())
        }
        (AnyArrayViewMut::F64(decoded), ZfpDType::F64) => {
            stream.decompress(mode, decoded.squeeze())
        }
        (decoded, dtype) => Err(ZfpCodecError::MismatchedDecodeIntoArray {
            source: AnyArrayAssignError::DTypeMismatch {
                src: dtype.into_dtype(),
                dst: decoded.dtype(),
            },
        }),
    }
}

#[derive(Serialize, Deserialize)]
struct CompressionHeader<'a> {
    dtype: ZfpDType,
    #[serde(borrow)]
    shape: Cow<'a, [usize]>,
}

/// Dtypes that Zfp can compress and decompress
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
#[expect(missing_docs)]
pub enum ZfpDType {
    #[serde(rename = "i32", alias = "int32")]
    I32,
    #[serde(rename = "i64", alias = "int64")]
    I64,
    #[serde(rename = "f32", alias = "float32")]
    F32,
    #[serde(rename = "f64", alias = "float64")]
    F64,
}

impl ZfpDType {
    /// Get the corresponding [`AnyArrayDType`]
    #[must_use]
    pub const fn into_dtype(self) -> AnyArrayDType {
        match self {
            Self::I32 => AnyArrayDType::I32,
            Self::I64 => AnyArrayDType::I64,
            Self::F32 => AnyArrayDType::F32,
            Self::F64 => AnyArrayDType::F64,
        }
    }
}

impl fmt::Display for ZfpDType {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.write_str(match self {
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::F32 => "f32",
            Self::F64 => "f64",
        })
    }
}
