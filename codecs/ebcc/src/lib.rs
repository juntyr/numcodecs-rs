//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.87.0-blue
//! [repo]: https://github.com/juntyr/numcodecs-rs
//!
//! [Latest Version]: https://img.shields.io/crates/v/numcodecs-ebcc
//! [crates.io]: https://crates.io/crates/numcodecs-ebcc
//!
//! [Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs-ebcc
//! [docs.rs]: https://docs.rs/numcodecs-ebcc/
//!
//! [Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/numcodecs-rs/numcodecs_ebcc
//!
//! EBCC codec implementation for the [`numcodecs`] API.

#![allow(clippy::multiple_crate_versions)] // embedded-io

#[cfg(test)]
use ::serde_json as _;

use std::borrow::Cow;

use ndarray::{Array, Array1, ArrayBase, ArrayViewMut, Axis, Data, DataMut, Dimension, IxDyn};
use num_traits::Float;
use numcodecs::{
    AnyArray, AnyArrayDType, AnyArrayView, AnyArrayViewMut, AnyCowArray, Codec, StaticCodec,
    StaticCodecConfig, StaticCodecVersion,
};
use schemars::{JsonSchema, Schema, SchemaGenerator, json_schema};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

type EbccCodecVersion = StaticCodecVersion<0, 1, 0>;

/// Codec providing compression using EBCC.
///
/// EBCC combines JPEG2000 compression with error-bounded residual compression.
///
/// Arrays that are higher-dimensional than 3D are encoded by compressing each
/// 3D slice with EBCC independently. Specifically, the array's shape is
/// interpreted as `[.., depth, height, width]`. If you want to compress 3D
/// slices along three different axes, you can swizzle the array axes
/// beforehand.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct EbccCodec {
    /// EBCC residual compression
    #[serde(flatten)]
    pub residual: EbccResidualType,
    /// JPEG2000 positive base compression ratio
    pub base_cr: Positive<f32>,
    /// The codec's encoding format version. Do not provide this parameter explicitly.
    #[serde(default, rename = "_version")]
    pub version: EbccCodecVersion,
}

/// Residual compression types supported by EBCC.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "residual")]
#[serde(deny_unknown_fields)]
pub enum EbccResidualType {
    #[serde(rename = "jpeg2000-only")]
    /// No residual compression - base JPEG2000 only
    Jpeg2000Only,
    #[serde(rename = "absolute")]
    /// Residual compression with absolute maximum error bound
    AbsoluteError {
        /// The positive maximum absolute error bound
        error: Positive<f32>,
    },
    #[serde(rename = "relative")]
    /// Residual compression with relative error bound
    RelativeError {
        /// The positive maximum relative error bound
        error: Positive<f32>,
    },
}

impl Codec for EbccCodec {
    type Error = EbccCodecError;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        match data {
            AnyCowArray::F32(data) => Ok(AnyArray::U8(
                Array1::from(compress(data, self.residual, self.base_cr)?).into_dyn(),
            )),
            encoded => Err(EbccCodecError::UnsupportedDtype(encoded.dtype())),
        }
    }

    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
        let AnyCowArray::U8(encoded) = encoded else {
            return Err(EbccCodecError::EncodedDataNotBytes {
                dtype: encoded.dtype(),
            });
        };

        if !matches!(encoded.shape(), [_]) {
            return Err(EbccCodecError::EncodedDataNotOneDimensional {
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
            return Err(EbccCodecError::EncodedDataNotBytes {
                dtype: encoded.dtype(),
            });
        };

        if !matches!(encoded.shape(), [_]) {
            return Err(EbccCodecError::EncodedDataNotOneDimensional {
                shape: encoded.shape().to_vec(),
            });
        }

        match decoded {
            AnyArrayViewMut::F32(decoded) => {
                decompress_into(&AnyArrayView::U8(encoded).as_bytes(), decoded)
            }
            decoded => Err(EbccCodecError::UnsupportedDtype(decoded.dtype())),
        }
    }
}

impl StaticCodec for EbccCodec {
    const CODEC_ID: &'static str = "ebcc.rs";

    type Config<'de> = Self;

    fn from_config(config: Self::Config<'_>) -> Self {
        config
    }

    fn get_config(&self) -> StaticCodecConfig<'_, Self> {
        StaticCodecConfig::from(self)
    }
}

/// Errors that may occur when applying the [`EbccCodec`].
#[derive(Debug, thiserror::Error)]
pub enum EbccCodecError {
    /// [`EbccCodec`] does not support the dtype
    #[error("Ebcc does not support the dtype {0}")]
    UnsupportedDtype(AnyArrayDType),
    /// [`EbccCodec`] failed to encode the header
    #[error("Ebcc failed to encode the header")]
    HeaderEncodeFailed {
        /// Opaque source error
        source: EbccHeaderError,
    },
    /// [`EbccCodec`] can only encode >2D data where the last two dimensions
    /// must be at least 32x32 but received an array with an insufficient shape
    #[error(
        "Ebcc can only encode >2D data where the last two dimensions must be at least 32x32 but received an array of shape {shape:?}"
    )]
    InsufficientDimensions {
        /// The unexpected shape of the array
        shape: Vec<usize>,
    },
    /// [`EbccCodec`] failed to encode the data
    #[error("Ebcc failed to encode the data")]
    EbccEncodeFailed {
        /// Opaque source error
        source: EbccCodingError,
    },
    /// [`EbccCodec`] failed to encode a 3D slice
    #[error("Ebcc failed to encode a 3D slice")]
    SliceEncodeFailed {
        /// Opaque source error
        source: EbccSliceError,
    },
    /// [`EbccCodec`] can only decode one-dimensional byte arrays but received
    /// an array of a different dtype
    #[error(
        "Ebcc can only decode one-dimensional byte arrays but received an array of dtype {dtype}"
    )]
    EncodedDataNotBytes {
        /// The unexpected dtype of the encoded array
        dtype: AnyArrayDType,
    },
    /// [`EbccCodec`] can only decode one-dimensional byte arrays but received
    /// an array of a different shape
    #[error(
        "Ebcc can only decode one-dimensional byte arrays but received a byte array of shape {shape:?}"
    )]
    EncodedDataNotOneDimensional {
        /// The unexpected shape of the encoded array
        shape: Vec<usize>,
    },
    /// [`EbccCodec`] failed to decode the header
    #[error("Ebcc failed to decode the header")]
    HeaderDecodeFailed {
        /// Opaque source error
        source: EbccHeaderError,
    },
    /// [`EbccCodec`] cannot decode into an array with a mismatching shape
    #[error("Ebcc cannot decode an array of shape {decoded:?} into an array of shape {array:?}")]
    DecodeIntoShapeMismatch {
        /// The shape of the decoded data
        decoded: Vec<usize>,
        /// The mismatching shape of the array to decode into
        array: Vec<usize>,
    },
    /// [`EbccCodec`] failed to decode a 3D slice
    #[error("Ebcc failed to decode a slice")]
    SliceDecodeFailed {
        /// Opaque source error
        source: EbccSliceError,
    },
    /// [`EbccCodec`] failed to decode from an excessive number of slices
    #[error("Ebcc failed to decode from an excessive number of slices")]
    DecodeTooManySlices,
    /// [`EbccCodec`] failed to decode the data
    #[error("Ebcc failed to decode the data")]
    EbccDecodeFailed {
        /// Opaque source error
        source: EbccCodingError,
    },
}

#[expect(clippy::derive_partial_eq_without_eq)] // floats are not Eq
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Hash)]
/// Positive floating point number
pub struct Positive<T: Float>(T);

impl<T: Float> PartialEq<T> for Positive<T> {
    fn eq(&self, other: &T) -> bool {
        self.0 == *other
    }
}

impl Serialize for Positive<f32> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_f32(self.0)
    }
}

impl<'de> Deserialize<'de> for Positive<f32> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let x = f32::deserialize(deserializer)?;

        if x > 0.0 {
            Ok(Self(x))
        } else {
            Err(serde::de::Error::invalid_value(
                serde::de::Unexpected::Float(f64::from(x)),
                &"a positive value",
            ))
        }
    }
}

impl JsonSchema for Positive<f32> {
    fn schema_name() -> Cow<'static, str> {
        Cow::Borrowed("PositiveF32")
    }

    fn schema_id() -> Cow<'static, str> {
        Cow::Borrowed(concat!(module_path!(), "::", "Positive<f32>"))
    }

    fn json_schema(_gen: &mut SchemaGenerator) -> Schema {
        json_schema!({
            "type": "number",
            "exclusiveMinimum": 0.0
        })
    }
}

#[derive(Debug, Error)]
#[error(transparent)]
/// Opaque error for when encoding or decoding the header fails
pub struct EbccHeaderError(postcard::Error);

#[derive(Debug, Error)]
#[error(transparent)]
/// Opaque error for when encoding or decoding a 3D slice fails
pub struct EbccSliceError(postcard::Error);

#[derive(Debug, Error)]
#[error(transparent)]
/// Opaque error for when encoding or decoding with EBCC fails
pub struct EbccCodingError(ebcc::EBCCError);

/// Compress the `data` array using EBCC with the provided `residual` and
/// `base_cr`.
///
/// # Errors
///
/// Errors with
/// - [`EbccCodecError::HeaderEncodeFailed`] if encoding the header failed
/// - [`EbccCodecError::InsufficientDimensions`] if the `data` has fewer than
///   two dimensions or the last two dimensions are not at least 32x32
/// - [`EbccCodecError::EbccEncodeFailed`] if encoding with EBCC failed
/// - [`EbccCodecError::SliceEncodeFailed`] if encoding a 3D slice failed
#[allow(clippy::missing_panics_doc)]
pub fn compress<S: Data<Elem = f32>, D: Dimension>(
    data: ArrayBase<S, D>,
    residual: EbccResidualType,
    base_cr: Positive<f32>,
) -> Result<Vec<u8>, EbccCodecError> {
    let mut encoded = postcard::to_extend(
        &CompressionHeader {
            dtype: EbccDType::F32,
            shape: Cow::Borrowed(data.shape()),
            version: StaticCodecVersion,
        },
        Vec::new(),
    )
    .map_err(|err| EbccCodecError::HeaderEncodeFailed {
        source: EbccHeaderError(err),
    })?;

    // EBCC cannot handle zero-length dimensions
    if data.is_empty() {
        return Ok(encoded);
    }

    let mut chunk_size = Vec::from(data.shape());
    let (width, height, depth) = match *chunk_size.as_mut_slice() {
        [ref mut rest @ .., depth, height, width] => {
            for r in rest {
                *r = 1;
            }
            (width, height, depth)
        }
        [height, width] => (width, height, 1),
        _ => {
            return Err(EbccCodecError::InsufficientDimensions {
                shape: Vec::from(data.shape()),
            });
        }
    };

    if (width < 32) || (height < 32) {
        return Err(EbccCodecError::InsufficientDimensions {
            shape: Vec::from(data.shape()),
        });
    }

    for mut slice in data.into_dyn().exact_chunks(chunk_size.as_slice()) {
        while slice.ndim() < 3 {
            slice = slice.insert_axis(Axis(0));
        }
        #[expect(clippy::unwrap_used)]
        // slice must now have at least three axes, and all but the last three
        //  must be of size 1
        let slice = slice.into_shape_with_order((depth, height, width)).unwrap();

        let encoded_slice = ebcc::ebcc_encode(
            slice,
            &ebcc::EBCCConfig {
                base_cr: base_cr.0,
                residual_compression_type: match residual {
                    EbccResidualType::Jpeg2000Only => ebcc::EBCCResidualType::Jpeg2000Only,
                    EbccResidualType::AbsoluteError { error } => {
                        ebcc::EBCCResidualType::AbsoluteError(error.0)
                    }
                    EbccResidualType::RelativeError { error } => {
                        ebcc::EBCCResidualType::RelativeError(error.0)
                    }
                },
            },
        )
        .map_err(|err| EbccCodecError::EbccEncodeFailed {
            source: EbccCodingError(err),
        })?;

        encoded = postcard::to_extend(encoded_slice.as_slice(), encoded).map_err(|err| {
            EbccCodecError::SliceEncodeFailed {
                source: EbccSliceError(err),
            }
        })?;
    }

    Ok(encoded)
}

/// Decompress the `encoded` data into an array using EBCC.
///
/// # Errors
///
/// Errors with
/// - [`EbccCodecError::HeaderDecodeFailed`] if decoding the header failed
/// - [`EbccCodecError::SliceDecodeFailed`] if decoding a 3D slice failed
/// - [`EbccCodecError::EbccDecodeFailed`] if decoding with EBCC failed
/// - [`EbccCodecError::DecodeTooManySlices`] if the encoded data contains
///   too many slices
pub fn decompress(encoded: &[u8]) -> Result<AnyArray, EbccCodecError> {
    fn decompress_typed(
        encoded: &[u8],
        shape: &[usize],
    ) -> Result<Array<f32, IxDyn>, EbccCodecError> {
        let mut decoded = Array::<f32, _>::zeros(shape);
        decompress_into_typed(encoded, decoded.view_mut())?;
        Ok(decoded)
    }

    let (header, encoded) =
        postcard::take_from_bytes::<CompressionHeader>(encoded).map_err(|err| {
            EbccCodecError::HeaderDecodeFailed {
                source: EbccHeaderError(err),
            }
        })?;

    // Return empty data for zero-size arrays
    if header.shape.iter().copied().any(|s| s == 0) {
        return match header.dtype {
            EbccDType::F32 => Ok(AnyArray::F32(Array::zeros(&*header.shape))),
        };
    }

    match header.dtype {
        EbccDType::F32 => Ok(AnyArray::F32(decompress_typed(encoded, &header.shape)?)),
    }
}

/// Decompress the `encoded` data into the `decoded` array using EBCC.
///
/// # Errors
///
/// Errors with
/// - [`EbccCodecError::HeaderDecodeFailed`] if decoding the header failed
/// - [`EbccCodecError::DecodeIntoShapeMismatch`] is the `decoded` array shape
///   does not match the shape of the decoded data
/// - [`EbccCodecError::SliceDecodeFailed`] if decoding a 3D slice failed
/// - [`EbccCodecError::EbccDecodeFailed`] if decoding with EBCC failed
/// - [`EbccCodecError::DecodeTooManySlices`] if the encoded data contains
///   too many slices
pub fn decompress_into<S: DataMut<Elem = f32>, D: Dimension>(
    encoded: &[u8],
    decoded: ArrayBase<S, D>,
) -> Result<(), EbccCodecError> {
    let (header, encoded) =
        postcard::take_from_bytes::<CompressionHeader>(encoded).map_err(|err| {
            EbccCodecError::HeaderDecodeFailed {
                source: EbccHeaderError(err),
            }
        })?;

    if decoded.shape() != &*header.shape {
        return Err(EbccCodecError::DecodeIntoShapeMismatch {
            decoded: header.shape.into_owned(),
            array: Vec::from(decoded.shape()),
        });
    }

    // Return empty data for zero-size arrays
    if header.shape.iter().copied().any(|s| s == 0) {
        return match header.dtype {
            EbccDType::F32 => Ok(()),
        };
    }

    match header.dtype {
        EbccDType::F32 => decompress_into_typed(encoded, decoded.into_dyn().view_mut()),
    }
}

fn decompress_into_typed(
    mut encoded: &[u8],
    mut decoded: ArrayViewMut<f32, IxDyn>,
) -> Result<(), EbccCodecError> {
    let mut chunk_size = Vec::from(decoded.shape());
    let (width, height, depth) = match *chunk_size.as_mut_slice() {
        [ref mut rest @ .., depth, height, width] => {
            for r in rest {
                *r = 1;
            }
            (width, height, depth)
        }
        [height, width] => (width, height, 1),
        [width] => (width, 1, 1),
        [] => (1, 1, 1),
    };

    for mut slice in decoded.exact_chunks_mut(chunk_size.as_slice()) {
        let (encoded_slice, rest) =
            postcard::take_from_bytes::<Cow<[u8]>>(encoded).map_err(|err| {
                EbccCodecError::SliceDecodeFailed {
                    source: EbccSliceError(err),
                }
            })?;
        encoded = rest;

        while slice.ndim() < 3 {
            slice = slice.insert_axis(Axis(0));
        }
        #[expect(clippy::unwrap_used)]
        // slice must now have at least three axes, and all but the last
        //  three must be of size 1
        let slice = slice.into_shape_with_order((depth, height, width)).unwrap();

        ebcc::ebcc_decode_into(&encoded_slice, slice).map_err(|err| {
            EbccCodecError::EbccDecodeFailed {
                source: EbccCodingError(err),
            }
        })?;
    }

    if !encoded.is_empty() {
        return Err(EbccCodecError::DecodeTooManySlices);
    }

    Ok(())
}

#[derive(Serialize, Deserialize)]
struct CompressionHeader<'a> {
    dtype: EbccDType,
    #[serde(borrow)]
    shape: Cow<'a, [usize]>,
    version: EbccCodecVersion,
}

/// Dtypes that EBCC can compress and decompress
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
enum EbccDType {
    #[serde(rename = "f32", alias = "float32")]
    F32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unsupported_dtype() {
        let codec = EbccCodec {
            residual: EbccResidualType::Jpeg2000Only,
            base_cr: Positive(10.0),
            version: StaticCodecVersion,
        };

        let data = Array1::<i32>::zeros(100);
        let result = codec.encode(AnyCowArray::I32(data.into_dyn().into()));

        assert!(matches!(result, Err(EbccCodecError::UnsupportedDtype(_))));
    }

    #[test]
    fn test_invalid_dimensions() {
        let codec = EbccCodec {
            residual: EbccResidualType::Jpeg2000Only,
            base_cr: Positive(10.0),
            version: StaticCodecVersion,
        };

        // Test dimensions too small (32 < 32x32 requirement)
        let data = Array::zeros(32);
        let result = codec.encode(AnyCowArray::F32(data.into_dyn().into()));
        assert!(
            matches!(result, Err(EbccCodecError::InsufficientDimensions { shape }) if shape == [32])
        );

        // Test dimensions too small (16x16 < 32x32 requirement)
        let data = Array::zeros((16, 16));
        let result = codec.encode(AnyCowArray::F32(data.into_dyn().into()));
        assert!(
            matches!(result, Err(EbccCodecError::InsufficientDimensions { shape }) if shape == [16, 16])
        );

        // Test mixed valid/invalid dimensions
        let data = Array::zeros((1, 32, 16));
        let result = codec.encode(AnyCowArray::F32(data.into_dyn().into()));
        assert!(
            matches!(result, Err(EbccCodecError::InsufficientDimensions { shape }) if shape == [1, 32, 16])
        );

        // Test valid dimensions
        let data = Array::zeros((1, 32, 32));
        let result = codec.encode(AnyCowArray::F32(data.into_dyn().into()));
        assert!(result.is_ok());

        // Test valid dimensions with slicing
        let data = Array::zeros((2, 2, 2, 32, 32));
        let result = codec.encode(AnyCowArray::F32(data.into_dyn().into()));
        assert!(result.is_ok());
    }

    #[test]
    fn test_large_array() -> Result<(), EbccCodecError> {
        // Test with a larger array (similar to small climate dataset)
        let height = 721; // Quarter degree resolution
        let width = 1440;
        let frames = 1;

        #[expect(clippy::suboptimal_flops, clippy::cast_precision_loss)]
        let data = Array::from_shape_fn((frames, height, width), |(_k, i, j)| {
            let lat = -90.0 + (i as f32 / height as f32) * 180.0;
            let lon = -180.0 + (j as f32 / width as f32) * 360.0;
            #[allow(clippy::let_and_return)]
            let temp = 273.15 + 30.0 * (1.0 - lat.abs() / 90.0) + 5.0 * (lon / 180.0).sin();
            temp
        });

        let codec_error = 0.1;
        let codec = EbccCodec {
            residual: EbccResidualType::AbsoluteError {
                error: Positive(codec_error),
            },
            base_cr: Positive(20.0),
            version: StaticCodecVersion,
        };

        let encoded = codec.encode(AnyArray::F32(data.clone().into_dyn()).into_cow())?;
        let decoded = codec.decode(encoded.cow())?;

        let AnyArray::U8(encoded) = encoded else {
            return Err(EbccCodecError::EncodedDataNotBytes {
                dtype: encoded.dtype(),
            });
        };

        let AnyArray::F32(decoded) = decoded else {
            return Err(EbccCodecError::UnsupportedDtype(decoded.dtype()));
        };

        // Check compression ratio
        let original_size = data.len() * std::mem::size_of::<f32>();
        #[allow(clippy::cast_precision_loss)]
        let compression_ratio = original_size as f64 / encoded.len() as f64;

        assert!(
            compression_ratio > 5.0,
            "Compression ratio {compression_ratio} should be at least 5:1",
        );

        // Check error bound is respected
        let max_error = data
            .iter()
            .zip(decoded.iter())
            .map(|(&orig, &decomp)| (orig - decomp).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_error <= (codec_error + 1e-6),
            "Max error {max_error} exceeds error bound {codec_error}",
        );

        Ok(())
    }
}
