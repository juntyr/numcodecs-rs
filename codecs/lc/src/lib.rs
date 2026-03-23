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

use std::{borrow::Cow, ffi::CString, io};

use ndarray::Array1;
use numcodecs::{
    AnyArray, AnyArrayAssignError, AnyArrayDType, AnyArrayView, AnyArrayViewMut, AnyCowArray,
    Codec, StaticCodec, StaticCodecConfig, StaticCodecVersion,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;

type LcCodecVersion = StaticCodecVersion<0, 1, 0>;

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
/// Codec providing compression using LC
pub struct LcCodec {
    /// LC preprocessor
    pub preprocessor: String,
    /// LC components
    pub components: String,
    /// The codec's encoding format version. Do not provide this parameter explicitly.
    #[serde(default, rename = "_version")]
    pub version: LcCodecVersion,
}

impl Codec for LcCodec {
    type Error = LcCodecError;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        compress(data.view(), &self.preprocessor, &self.components)
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
            &self.preprocessor,
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
            &self.preprocessor,
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
pub struct LcCodingError(io::Error);

#[expect(clippy::needless_pass_by_value)]
/// Compress the `array` using LC with the provided `preprocessor` and
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
    preprocessor: &str,
    components: &str,
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

    let preprocessor = CString::new(preprocessor).unwrap();
    let components = CString::new(components).unwrap();

    encoded.append(
        &mut lc_framework::compress(&preprocessor, &components, &*array.as_bytes()).map_err(
            |()| LcCodecError::LcEncodeFailed {
                source: LcCodingError(io::Error::other("todo")),
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
    preprocessor: &str,
    components: &str,
    encoded: &[u8],
) -> Result<AnyArray, LcCodecError> {
    let (header, encoded) =
        postcard::take_from_bytes::<CompressionHeader>(encoded).map_err(|err| {
            LcCodecError::HeaderDecodeFailed {
                source: LcHeaderError(err),
            }
        })?;

    let (decoded, result) = AnyArray::with_zeros_bytes(header.dtype, &header.shape, |decoded| {
        decompress_into_bytes(preprocessor, components, encoded, decoded)
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
    preprocessor: &str,
    components: &str,
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

    decoded
        .with_bytes_mut(|decoded| decompress_into_bytes(preprocessor, components, encoded, decoded))
}

fn decompress_into_bytes(
    preprocessor: &str,
    components: &str,
    encoded: &[u8],
    decoded: &mut [u8],
) -> Result<(), LcCodecError> {
    // LC does not support empty input, so skip decoding
    if decoded.is_empty() && encoded.is_empty() {
        return Ok(());
    }

    let preprocessor = CString::new(preprocessor).unwrap();
    let components = CString::new(components).unwrap();

    let dec = lc_framework::decompress(&preprocessor, &components, encoded).map_err(|()| {
        LcCodecError::LcDecodeFailed {
            source: LcCodingError(io::Error::other("todod")),
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

        let preprocessor = "";
        let components = "BIT_4 RLE_4";

        let compressed = compress(
            AnyArrayView::F32(data.view().into_dyn()),
            preprocessor,
            components,
        )
        .unwrap();
        let decompressed = decompress(preprocessor, components, &compressed).unwrap();

        assert_eq!(decompressed, AnyArray::F32(data.into_dyn()));
    }

    #[test]
    fn abs_error() {
        let data = ndarray::linspace(0.0, std::f32::consts::PI, 100)
            .collect::<Array1<f32>>()
            .into_shape_with_order((10, 10))
            .unwrap()
            .cos();

        let preprocessor = "QUANT_ABS_0_f32(0.1)";
        let components = "BIT_4 RLE_4";

        let compressed = compress(
            AnyArrayView::F32(data.view().into_dyn()),
            preprocessor,
            components,
        )
        .unwrap();
        let decompressed = decompress(preprocessor, components, &compressed).unwrap();

        let AnyArray::F32(decompressed) = decompressed else {
            panic!("unexpected decompressed dtype {}", decompressed.dtype());
        };
        assert_eq!(decompressed.shape(), data.shape());

        for (o, d) in data.into_iter().zip(decompressed) {
            assert!((o - d).abs() <= 0.1);
        }
    }
}
