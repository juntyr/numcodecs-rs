//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.82.0-blue
//! [repo]: https://github.com/juntyr/numcodecs-rs
//!
//! [Latest Version]: https://img.shields.io/crates/v/numcodecs-zfp-classic
//! [crates.io]: https://crates.io/crates/numcodecs-zfp-classic
//!
//! [Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs-zfp-classic
//! [docs.rs]: https://docs.rs/numcodecs-zfp-classic/
//!
//! [Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/numcodecs-rs/numcodecs_zfp_classic
//!
//! ZFP (classic) codec implementation for the [`numcodecs`] API.
//!
//! This implementation uses ZFP's default
//! [`ZFP_ROUNDING_MODE=ZFP_ROUND_NEVER`](https://zfp.readthedocs.io/en/release1.0.1/installation.html#c.ZFP_ROUNDING_MODE)
//! rounding mode, which is known to increase bias and correlation in ZFP's
//! errors
//! (see <https://zfp.readthedocs.io/en/release1.0.1/faq.html#zfp-rounding>).
//!
//! Please see the `numcodecs-zfp` codec for an implementation that uses an
//! improved version of ZFP.

#![allow(clippy::multiple_crate_versions)] // embedded-io

use std::{borrow::Cow, fmt};

use ndarray::{Array, Array1, ArrayView, Dimension};
use numcodecs::{
    AnyArray, AnyArrayAssignError, AnyArrayDType, AnyArrayView, AnyArrayViewMut, AnyCowArray,
    Codec, StaticCodec, StaticCodecConfig, StaticCodecVersion,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[cfg(test)]
use ::serde_json as _;

mod ffi;

type ZfpClassicCodecVersion = StaticCodecVersion<0, 1, 0>;

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
// serde cannot deny unknown fields because of the flatten
#[schemars(deny_unknown_fields)]
/// Codec providing compression using ZFP (classic)
pub struct ZfpClassicCodec {
    /// ZFP compression mode
    #[serde(flatten)]
    pub mode: ZfpCompressionMode,
    /// The codec's version. Do not provide this parameter explicitly.
    #[serde(default, rename = "_version")]
    pub version: ZfpClassicCodecVersion,
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

impl Codec for ZfpClassicCodec {
    type Error = ZfpClassicCodecError;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        if matches!(data.dtype(), AnyArrayDType::I32 | AnyArrayDType::I64)
            && matches!(
                self.mode,
                ZfpCompressionMode::FixedAccuracy { tolerance: _ }
            )
        {
            return Err(ZfpClassicCodecError::FixedAccuracyModeIntegerData);
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
            encoded => Err(ZfpClassicCodecError::UnsupportedDtype(encoded.dtype())),
        }
    }

    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
        let AnyCowArray::U8(encoded) = encoded else {
            return Err(ZfpClassicCodecError::EncodedDataNotBytes {
                dtype: encoded.dtype(),
            });
        };

        if !matches!(encoded.shape(), [_]) {
            return Err(ZfpClassicCodecError::EncodedDataNotOneDimensional {
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
            return Err(ZfpClassicCodecError::EncodedDataNotBytes {
                dtype: encoded.dtype(),
            });
        };

        if !matches!(encoded.shape(), [_]) {
            return Err(ZfpClassicCodecError::EncodedDataNotOneDimensional {
                shape: encoded.shape().to_vec(),
            });
        }

        decompress_into(&AnyArrayView::U8(encoded).as_bytes(), decoded)
    }
}

impl StaticCodec for ZfpClassicCodec {
    const CODEC_ID: &'static str = "zfp-classic.rs";

    type Config<'de> = Self;

    fn from_config(config: Self::Config<'_>) -> Self {
        config
    }

    fn get_config(&self) -> StaticCodecConfig<Self> {
        StaticCodecConfig::from(self)
    }
}

#[derive(Debug, Error)]
/// Errors that may occur when applying the [`ZfpClassicCodec`].
pub enum ZfpClassicCodecError {
    /// [`ZfpClassicCodec`] does not support the dtype
    #[error("ZfpClassic does not support the dtype {0}")]
    UnsupportedDtype(AnyArrayDType),
    /// [`ZfpClassicCodec`] does not support the fixed accuracy mode for
    /// integer data
    #[error("ZfpClassic does not support the fixed accuracy mode for integer data")]
    FixedAccuracyModeIntegerData,
    /// [`ZfpClassicCodec`] only supports 1-4 dimensional data
    #[error("ZfpClassic only supports 1-4 dimensional data but found shape {shape:?}")]
    ExcessiveDimensionality {
        /// The unexpected shape of the data
        shape: Vec<usize>,
    },
    /// [`ZfpClassicCodec`] was configured with an invalid expert `mode`
    #[error("ZfpClassic was configured with an invalid expert mode {mode:?}")]
    InvalidExpertMode {
        /// The unexpected compression mode
        mode: ZfpCompressionMode,
    },
    /// [`ZfpClassicCodec`] failed to encode the header
    #[error("ZfpClassic failed to encode the header")]
    HeaderEncodeFailed,
    /// [`ZfpClassicCodec`] failed to encode the array metadata header
    #[error("ZfpClassic failed to encode the array metadata header")]
    MetaHeaderEncodeFailed {
        /// Opaque source error
        source: ZfpHeaderError,
    },
    /// [`ZfpClassicCodec`] failed to encode the data
    #[error("ZfpClassic failed to encode the data")]
    ZfpEncodeFailed,
    /// [`ZfpClassicCodec`] can only decode one-dimensional byte arrays but
    /// received an array of a different dtype
    #[error(
        "ZfpClassic can only decode one-dimensional byte arrays but received an array of dtype {dtype}"
    )]
    EncodedDataNotBytes {
        /// The unexpected dtype of the encoded array
        dtype: AnyArrayDType,
    },
    /// [`ZfpClassicCodec`] can only decode one-dimensional byte arrays but
    /// received an array of a different shape
    #[error("ZfpClassic can only decode one-dimensional byte arrays but received a byte array of shape {shape:?}")]
    EncodedDataNotOneDimensional {
        /// The unexpected shape of the encoded array
        shape: Vec<usize>,
    },
    /// [`ZfpClassicCodec`] failed to decode the header
    #[error("ZfpClassic failed to decode the header")]
    HeaderDecodeFailed,
    /// [`ZfpClassicCodec`] failed to decode the array metadata header
    #[error("ZfpClassic failed to decode the array metadata header")]
    MetaHeaderDecodeFailed {
        /// Opaque source error
        source: ZfpHeaderError,
    },
    /// [`ZfpClassicCodec`] cannot decode into the provided array
    #[error("ZfpClassicCodec cannot decode into the provided array")]
    MismatchedDecodeIntoArray {
        /// The source of the error
        #[from]
        source: AnyArrayAssignError,
    },
    /// [`ZfpClassicCodec`] failed to decode the data
    #[error("ZfpClassic failed to decode the data")]
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
/// - [`ZfpClassicCodecError::ExcessiveDimensionality`] if data is more than
///   4-dimensional
/// - [`ZfpClassicCodecError::InvalidExpertMode`] if the `mode` has invalid
///   expert mode parameters
/// - [`ZfpClassicCodecError::HeaderEncodeFailed`] if encoding the ZFP header
///   failed
/// - [`ZfpClassicCodecError::MetaHeaderEncodeFailed`] if encoding the array
///   metadata header failed
/// - [`ZfpClassicCodecError::ZfpEncodeFailed`] if an opaque encoding error
///   occurred
pub fn compress<T: ffi::ZfpCompressible, D: Dimension>(
    data: ArrayView<T, D>,
    mode: &ZfpCompressionMode,
) -> Result<Vec<u8>, ZfpClassicCodecError> {
    let mut encoded = postcard::to_extend(
        &CompressionHeader {
            dtype: <T as ffi::ZfpCompressible>::D_TYPE,
            shape: Cow::Borrowed(data.shape()),
            version: StaticCodecVersion,
        },
        Vec::new(),
    )
    .map_err(|err| ZfpClassicCodecError::MetaHeaderEncodeFailed {
        source: ZfpHeaderError(err),
    })?;

    // ZFP cannot handle zero-length dimensions
    if data.is_empty() {
        return Ok(encoded);
    }

    // Setup zfp structs to begin compression
    // Squeeze the data to avoid wasting ZFP dimensions on axes of length 1
    let field = ffi::ZfpField::new(data.into_dyn().squeeze())?;
    let stream = ffi::ZfpCompressionStream::new(&field, mode)?;

    // Allocate space based on the maximum size potentially required by zfp to
    //  store the compressed array
    let stream = stream.with_bitstream(field, &mut encoded);

    // Write the header so we can reconstruct ZFP's mode on decompression
    let stream = stream.write_header()?;

    // Compress the field into the allocated output array
    stream.compress()?;

    Ok(encoded)
}

/// Decompress the `encoded` data into an array using ZFP.
///
/// # Errors
///
/// Errors with
/// - [`ZfpClassicCodecError::HeaderDecodeFailed`] if decoding the ZFP header
///   failed
/// - [`ZfpClassicCodecError::MetaHeaderDecodeFailed`] if decoding the array
///   metadata header failed
/// - [`ZfpClassicCodecError::ZfpDecodeFailed`] if an opaque decoding error
///   occurred
pub fn decompress(encoded: &[u8]) -> Result<AnyArray, ZfpClassicCodecError> {
    let (header, encoded) =
        postcard::take_from_bytes::<CompressionHeader>(encoded).map_err(|err| {
            ZfpClassicCodecError::MetaHeaderDecodeFailed {
                source: ZfpHeaderError(err),
            }
        })?;

    // Return empty data for zero-size arrays
    if header.shape.iter().copied().product::<usize>() == 0 {
        let decoded = match header.dtype {
            ZfpDType::I32 => AnyArray::I32(Array::zeros(&*header.shape)),
            ZfpDType::I64 => AnyArray::I64(Array::zeros(&*header.shape)),
            ZfpDType::F32 => AnyArray::F32(Array::zeros(&*header.shape)),
            ZfpDType::F64 => AnyArray::F64(Array::zeros(&*header.shape)),
        };
        return Ok(decoded);
    }

    // Setup zfp structs to begin decompression
    let stream = ffi::ZfpDecompressionStream::new(encoded);

    // Read the header to reconstruct ZFP's mode
    let stream = stream.read_header()?;

    // Decompress the field into a newly allocated output array
    match header.dtype {
        ZfpDType::I32 => {
            let mut decompressed = Array::zeros(&*header.shape);
            stream.decompress_into(decompressed.view_mut().squeeze())?;
            Ok(AnyArray::I32(decompressed))
        }
        ZfpDType::I64 => {
            let mut decompressed = Array::zeros(&*header.shape);
            stream.decompress_into(decompressed.view_mut().squeeze())?;
            Ok(AnyArray::I64(decompressed))
        }
        ZfpDType::F32 => {
            let mut decompressed = Array::zeros(&*header.shape);
            stream.decompress_into(decompressed.view_mut().squeeze())?;
            Ok(AnyArray::F32(decompressed))
        }
        ZfpDType::F64 => {
            let mut decompressed = Array::zeros(&*header.shape);
            stream.decompress_into(decompressed.view_mut().squeeze())?;
            Ok(AnyArray::F64(decompressed))
        }
    }
}

/// Decompress the `encoded` data into a `decoded` array using ZFP.
///
/// # Errors
///
/// Errors with
/// - [`ZfpClassicCodecError::HeaderDecodeFailed`] if decoding the ZFP header
///   failed
/// - [`ZfpClassicCodecError::MetaHeaderDecodeFailed`] if decoding the array
///   metadata header failed
/// - [`ZfpClassicCodecError::MismatchedDecodeIntoArray`] if the `decoded`
///   array is of the wrong dtype or shape
/// - [`ZfpClassicCodecError::ZfpDecodeFailed`] if an opaque decoding error
///   occurred
pub fn decompress_into(
    encoded: &[u8],
    decoded: AnyArrayViewMut,
) -> Result<(), ZfpClassicCodecError> {
    let (header, encoded) =
        postcard::take_from_bytes::<CompressionHeader>(encoded).map_err(|err| {
            ZfpClassicCodecError::MetaHeaderDecodeFailed {
                source: ZfpHeaderError(err),
            }
        })?;

    if decoded.shape() != &*header.shape {
        return Err(ZfpClassicCodecError::MismatchedDecodeIntoArray {
            source: AnyArrayAssignError::ShapeMismatch {
                src: header.shape.into_owned(),
                dst: decoded.shape().to_vec(),
            },
        });
    }

    // Empty data doesn't need to be initialized
    if decoded.is_empty() {
        return Ok(());
    }

    // Setup zfp structs to begin decompression
    let stream = ffi::ZfpDecompressionStream::new(encoded);

    // Read the header to reconstruct ZFP's mode
    let stream = stream.read_header()?;

    // Decompress the field into the output array
    match (decoded, header.dtype) {
        (AnyArrayViewMut::I32(decoded), ZfpDType::I32) => stream.decompress_into(decoded.squeeze()),
        (AnyArrayViewMut::I64(decoded), ZfpDType::I64) => stream.decompress_into(decoded.squeeze()),
        (AnyArrayViewMut::F32(decoded), ZfpDType::F32) => stream.decompress_into(decoded.squeeze()),
        (AnyArrayViewMut::F64(decoded), ZfpDType::F64) => stream.decompress_into(decoded.squeeze()),
        (decoded, dtype) => Err(ZfpClassicCodecError::MismatchedDecodeIntoArray {
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
    version: ZfpClassicCodecVersion,
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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use ndarray::ArrayView1;

    use super::*;

    #[test]
    fn zero_length() {
        let encoded = compress(
            Array::<f32, _>::from_shape_vec([1, 27, 0].as_slice(), vec![])
                .unwrap()
                .view(),
            &ZfpCompressionMode::FixedPrecision { precision: 7 },
        )
        .unwrap();
        let decoded = decompress(&encoded).unwrap();

        assert_eq!(decoded.dtype(), AnyArrayDType::F32);
        assert!(decoded.is_empty());
        assert_eq!(decoded.shape(), &[1, 27, 0]);
    }

    #[test]
    fn one_dimension() {
        let data = Array::from_shape_vec(
            [2_usize, 1, 2, 1, 1, 1].as_slice(),
            vec![1.0, 2.0, 3.0, 4.0],
        )
        .unwrap();

        let encoded = compress(
            data.view(),
            &ZfpCompressionMode::FixedAccuracy { tolerance: 0.1 },
        )
        .unwrap();
        let decoded = decompress(&encoded).unwrap();

        assert_eq!(decoded, AnyArray::F32(data));
    }

    #[test]
    fn small_state() {
        for data in [
            &[][..],
            &[0.0],
            &[0.0, 1.0],
            &[0.0, 1.0, 0.0],
            &[0.0, 1.0, 0.0, 1.0],
        ] {
            let encoded = compress(
                ArrayView1::from(data),
                &ZfpCompressionMode::FixedAccuracy { tolerance: 0.1 },
            )
            .unwrap();
            let decoded = decompress(&encoded).unwrap();

            assert_eq!(
                decoded,
                AnyArray::F64(Array1::from_vec(data.to_vec()).into_dyn())
            );
        }
    }
}
