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

use ndarray::{Array1, ArrayView, Dimension};
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
    /// In fixed-rate mode, each d-dimensional compressed block of 4^d values
    /// is stored using a fixed number of bits. This number of compressed bits
    /// per block is amortized over the 4^d values to give a rate of
    /// `rate = max_bits / 4^d` in bits per value.
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
    /// number is chosen such that `minexp = floor(log2(tolerance))`.
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

        decompress(&AnyCowArray::U8(encoded).as_bytes())
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

        decompress_into(&AnyArrayView::U8(encoded).as_bytes(), decoded)
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
    /// [`ZfpCodec`] failed to encode the header
    #[error("Zfp failed to encode the header")]
    HeaderEncodeFailed,
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
    HeaderDecodeFailed,
    /// [`ZfpCodec`] cannot decode into the provided array
    #[error("ZfpCodec cannot decode into the provided array")]
    MismatchedDecodeIntoArray {
        /// The source of the error
        #[from]
        source: AnyArrayAssignError,
    },
    /// [`ZfpCodec`] failed to decode the data with the unknown dtype
    #[error("Zfp failed to decode the data with an unknown dtype #{0}")]
    DecodeUnknownDtype(u32),
    /// [`ZfpCodec`] failed to decode the data
    #[error("Zfp failed to decode the data")]
    ZfpDecodeFailed,
}

/// Compress the `data` array using ZFP with the provided `mode`.
///
/// # Errors
///
/// Errors with
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
    // Setup zfp structs to begin compression
    let field = ffi::ZfpField::new(data)?;
    let stream = ffi::ZfpCompressionStream::new(&field, mode)?;

    // Allocate space based on the maximum size potentially required by zfp to
    //  store the compressed array
    let stream = stream.with_bitstream(field);

    // Write the full header so we can reconstruct the array on decompression
    let stream = stream.write_full_header()?;

    // Compress the field into the allocated output array
    stream.compress()
}

/// Decompress the `encoded` data into an array using ZFP.
///
/// # Errors
///
/// Errors with
/// - [`ZfpCodecError::HeaderDecodeFailed`] if decoding the header failed
/// - [`ZfpCodecError::DecodeUnknownDtype`] if the encoded data uses an unknown
///   dtype
/// - [`ZfpCodecError::ZfpDecodeFailed`] if an opaque decoding error occurred
pub fn decompress(encoded: &[u8]) -> Result<AnyArray, ZfpCodecError> {
    // Setup zfp structs to begin decompression
    let stream = ffi::ZfpDecompressionStream::new(encoded);

    // Read the full header to verify the decompression dtype
    let stream = stream.read_full_header()?;

    // Decompress the field into a newly allocated output array
    stream.decompress()
}

/// Decompress the `encoded` data into a `decoded` array using ZFP.
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
pub fn decompress_into(encoded: &[u8], decoded: AnyArrayViewMut) -> Result<(), ZfpCodecError> {
    // Setup zfp structs to begin decompression
    let stream = ffi::ZfpDecompressionStream::new(encoded);

    // Read the full header to verify the decompression dtype
    let stream = stream.read_full_header()?;

    stream.decompress_into(decoded)
}
