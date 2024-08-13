//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.76.0-blue
//! [repo]: https://github.com/juntyr/numcodecs-rs
//!
//! [Latest Version]: https://img.shields.io/crates/v/numcodecs-zstd
//! [crates.io]: https://crates.io/crates/numcodecs-zstd
//!
//! [Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs-zstd
//! [docs.rs]: https://docs.rs/numcodecs-zstd/
//!
//! [Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/numcodecs-rs/numcodecs_zstd
//!
//! Zstandard codec implementation for the [`numcodecs`] API.

// Only used to explicitly enable the `no_wasm_shim` feature in zstd/zstd-sys
use zstd_sys as _;

use std::{borrow::Cow, io};

use ndarray::Array1;
use numcodecs::{
    AnyArray, AnyArrayDType, AnyArrayView, AnyArrayViewMut, AnyCowArray, Codec, StaticCodec,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

#[derive(Clone, Serialize, Deserialize)]
/// Codec providing compression using Zstandard
pub struct ZstdCodec {
    /// Compression level
    pub level: ZstdLevel,
}

impl Codec for ZstdCodec {
    type Error = ZstdCodecError;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        compress(data.view(), self.level)
            .map(|bytes| AnyArray::U8(Array1::from_vec(bytes).into_dyn()))
    }

    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
        let AnyCowArray::U8(encoded) = encoded else {
            return Err(ZstdCodecError::EncodedDataNotBytes {
                dtype: encoded.dtype(),
            });
        };

        if !matches!(encoded.shape(), [_]) {
            return Err(ZstdCodecError::EncodedDataNotOneDimensional {
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
            return Err(ZstdCodecError::EncodedDataNotBytes {
                dtype: encoded.dtype(),
            });
        };

        if !matches!(encoded.shape(), [_]) {
            return Err(ZstdCodecError::EncodedDataNotOneDimensional {
                shape: encoded.shape().to_vec(),
            });
        }

        decompress_into(&AnyArrayView::U8(encoded).as_bytes(), decoded)
    }

    fn get_config<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.serialize(serializer)
    }
}

impl StaticCodec for ZstdCodec {
    const CODEC_ID: &'static str = "zstd";

    fn from_config<'de, D: Deserializer<'de>>(config: D) -> Result<Self, D::Error> {
        Self::deserialize(config)
    }
}

#[derive(Clone, Copy)]
/// Zstandard compression level.
///
/// The level ranges from small (fastest) to large (best compression).
pub struct ZstdLevel {
    level: zstd::zstd_safe::CompressionLevel,
}

impl serde::Serialize for ZstdLevel {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.level.serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for ZstdLevel {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let level = serde::Deserialize::deserialize(deserializer)?;

        let level_range = zstd::compression_level_range();

        if !level_range.contains(&level) {
            return Err(serde::de::Error::custom(format!(
                "level {level} is not in {}..={}",
                level_range.start(),
                level_range.end()
            )));
        }

        Ok(Self { level })
    }
}

#[derive(Debug, Error)]
/// Errors that may occur when applying the [`ZstdCodec`].
pub enum ZstdCodecError {
    /// [`ZstdCodec`] failed to encode the header
    #[error("Zstd failed to encode the header")]
    HeaderEncodeFailed {
        /// Opaque source error
        source: ZstdHeaderError,
    },
    /// [`ZstdCodec`] failed to encode the encoded data
    #[error("Zstd failed to decode the encoded data")]
    ZstdEncodeFailed {
        /// Opaque source error
        source: ZstdCodingError,
    },
    /// [`ZstdCodec`] can only decode one-dimensional byte arrays but received
    /// an array of a different dtype
    #[error(
        "Zstd can only decode one-dimensional byte arrays but received an array of dtype {dtype}"
    )]
    EncodedDataNotBytes {
        /// The unexpected dtype of the encoded array
        dtype: AnyArrayDType,
    },
    /// [`ZstdCodec`] can only decode one-dimensional byte arrays but received
    /// an array of a different shape
    #[error("Zstd can only decode one-dimensional byte arrays but received a byte array of shape {shape:?}")]
    EncodedDataNotOneDimensional {
        /// The unexpected shape of the encoded array
        shape: Vec<usize>,
    },
    /// [`ZstdCodec`] failed to encode the header
    #[error("Zstd failed to decode the header")]
    HeaderDecodeFailed {
        /// Opaque source error
        source: ZstdHeaderError,
    },
    /// [`ZstdCodec`] decode consumed less encoded data, which contains trailing
    /// junk
    #[error("Zstd decode consumed less encoded data, which contains trailing junk")]
    DecodeExcessiveEncodedData,
    /// [`ZstdCodec`] produced less decoded data than expected
    #[error("Zstd produced less decoded data than expected")]
    DecodeProducedLess,
    /// [`ZstdCodec`] failed to decode the encoded data
    #[error("Zstd failed to decode the encoded data")]
    ZstdDecodeFailed {
        /// Opaque source error
        source: ZstdCodingError,
    },
    /// [`ZstdCodec`] cannot decode the `decoded` dtype into the `provided`
    /// array
    #[error("Zstd cannot decode the dtype {decoded} into the provided {provided} array")]
    MismatchedDecodeIntoDtype {
        /// Dtype of the `decoded` data
        decoded: AnyArrayDType,
        /// Dtype of the `provided` array into which the data is to be decoded
        provided: AnyArrayDType,
    },
    /// [`ZstdCodec`] cannot decode the decoded array into the provided
    /// array of a different shape
    #[error("Zstd cannot decode the decoded array of shape {decoded:?} into the provided array of shape {provided:?}")]
    MismatchedDecodeIntoShape {
        /// Shape of the `decoded` data
        decoded: Vec<usize>,
        /// Shape of the `provided` array into which the data is to be decoded
        provided: Vec<usize>,
    },
}

#[derive(Debug, Error)]
#[error(transparent)]
/// Opaque error for when encoding or decoding the header fails
pub struct ZstdHeaderError(postcard::Error);

#[derive(Debug, Error)]
#[error(transparent)]
/// Opaque error for when encoding or decoding with Zstandard fails
pub struct ZstdCodingError(io::Error);

#[allow(clippy::needless_pass_by_value)]
/// Compress the `array` using Zstandard with the provided `level`.
///
/// # Errors
///
/// Errors with
/// - [`ZstdCodecError::HeaderEncodeFailed`] if encoding the header to the
///   output bytevec failed
/// - [`ZstdCodecError::ZstdEncodeFailed`] if an opaque encoding error occurred
///
/// # Panics
///
/// Panics if the infallible encoding with Zstd fails.
pub fn compress(array: AnyArrayView, level: ZstdLevel) -> Result<Vec<u8>, ZstdCodecError> {
    let mut encoded = postcard::to_extend(
        &CompressionHeader {
            dtype: array.dtype(),
            shape: Cow::Borrowed(array.shape()),
        },
        Vec::new(),
    )
    .map_err(|err| ZstdCodecError::HeaderEncodeFailed {
        source: ZstdHeaderError(err),
    })?;

    zstd::stream::copy_encode(&*array.as_bytes(), &mut encoded, level.level).map_err(|err| {
        ZstdCodecError::ZstdEncodeFailed {
            source: ZstdCodingError(err),
        }
    })?;

    Ok(encoded)
}

/// Decompress the `encoded` data into an array using Zstandard.
///
/// # Errors
///
/// Errors with
/// - [`ZstdCodecError::HeaderDecodeFailed`] if decoding the header failed
/// - [`ZstdCodecError::DecodeExcessiveEncodedData`] if the encoded data
///   contains excessive trailing data junk
/// - [`ZstdCodecError::DecodeProducedLess`] if decoding produced less data than
///   expected
/// - [`ZstdCodecError::ZstdDecodeFailed`] if an opaque decoding error occurred
pub fn decompress(encoded: &[u8]) -> Result<AnyArray, ZstdCodecError> {
    let (header, encoded) =
        postcard::take_from_bytes::<CompressionHeader>(encoded).map_err(|err| {
            ZstdCodecError::HeaderDecodeFailed {
                source: ZstdHeaderError(err),
            }
        })?;

    let (decoded, result) = AnyArray::with_zeros_bytes(header.dtype, &header.shape, |decoded| {
        decompress_into_bytes(encoded, decoded)
    });

    result.map(|()| decoded)
}

/// Decompress the `encoded` data into a `decoded` array using Zstandard.
///
/// # Errors
///
/// Errors with
/// - [`ZstdCodecError::HeaderDecodeFailed`] if decoding the header failed
/// - [`ZstdCodecError::MismatchedDecodeIntoDtype`] if the `decoded` array is of
///   the wrong dtype
/// - [`ZstdCodecError::MismatchedDecodeIntoShape`] if the `decoded` array is of
///   the wrong shape
/// - [`ZstdCodecError::HeaderDecodeFailed`] if decoding the header failed
/// - [`ZstdCodecError::DecodeExcessiveEncodedData`] if the encoded data
///   contains excessive trailing data junk
/// - [`ZstdCodecError::DecodeProducedLess`] if decoding produced less data than
///   expected
/// - [`ZstdCodecError::ZstdDecodeFailed`] if an opaque decoding error occurred
pub fn decompress_into(encoded: &[u8], mut decoded: AnyArrayViewMut) -> Result<(), ZstdCodecError> {
    let (header, encoded) =
        postcard::take_from_bytes::<CompressionHeader>(encoded).map_err(|err| {
            ZstdCodecError::HeaderDecodeFailed {
                source: ZstdHeaderError(err),
            }
        })?;

    if header.dtype != decoded.dtype() {
        return Err(ZstdCodecError::MismatchedDecodeIntoDtype {
            decoded: header.dtype,
            provided: decoded.dtype(),
        });
    }

    if header.shape != decoded.shape() {
        return Err(ZstdCodecError::MismatchedDecodeIntoShape {
            decoded: header.shape.into_owned(),
            provided: decoded.shape().to_vec(),
        });
    }

    decoded.with_bytes_mut(|decoded| decompress_into_bytes(encoded, decoded))
}

fn decompress_into_bytes(mut encoded: &[u8], mut decoded: &mut [u8]) -> Result<(), ZstdCodecError> {
    #[allow(clippy::needless_borrows_for_generic_args)]
    // we want to check encoded and decoded for full consumption after the decoding
    zstd::stream::copy_decode(&mut encoded, &mut decoded).map_err(|err| {
        ZstdCodecError::ZstdDecodeFailed {
            source: ZstdCodingError(err),
        }
    })?;

    if !encoded.is_empty() {
        return Err(ZstdCodecError::DecodeExcessiveEncodedData);
    }

    if !decoded.is_empty() {
        return Err(ZstdCodecError::DecodeProducedLess);
    }

    Ok(())
}

#[derive(serde::Serialize, serde::Deserialize)]
struct CompressionHeader<'a> {
    dtype: AnyArrayDType,
    #[serde(borrow)]
    shape: Cow<'a, [usize]>,
}
