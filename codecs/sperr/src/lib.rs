//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.87.0-blue
//! [repo]: https://github.com/juntyr/numcodecs-rs
//!
//! [Latest Version]: https://img.shields.io/crates/v/numcodecs-sperr
//! [crates.io]: https://crates.io/crates/numcodecs-sperr
//!
//! [Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs-sperr
//! [docs.rs]: https://docs.rs/numcodecs-sperr/
//!
//! [Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/numcodecs-rs/numcodecs_sperr
//!
//! SPERR codec implementation for the [`numcodecs`] API.

#![allow(clippy::multiple_crate_versions)] // embedded-io

#[cfg(test)]
use ::serde_json as _;

use std::borrow::Cow;
use std::fmt;

use ndarray::{Array, Array1, ArrayBase, Axis, Data, Dimension, IxDyn, ShapeError};
use num_traits::{Float, identities::Zero};
use numcodecs::{
    AnyArray, AnyArrayAssignError, AnyArrayDType, AnyArrayView, AnyArrayViewMut, AnyCowArray,
    Codec, StaticCodec, StaticCodecConfig, StaticCodecVersion,
};
use schemars::{JsonSchema, Schema, SchemaGenerator, json_schema};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

type SperrCodecVersion = StaticCodecVersion<0, 1, 0>;

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
// serde cannot deny unknown fields because of the flatten
#[schemars(deny_unknown_fields)]
/// Codec providing compression using SPERR.
///
/// Arrays that are higher-dimensional than 3D are encoded by compressing each
/// 3D slice with SPERR independently. Specifically, the array's shape is
/// interpreted as `[.., depth, height, width]`. If you want to compress 3D
/// slices along three different axes, you can swizzle the array axes
/// beforehand.
pub struct SperrCodec {
    /// SPERR compression mode
    #[serde(flatten)]
    pub mode: SperrCompressionMode,
    /// The codec's encoding format version. Do not provide this parameter explicitly.
    #[serde(default, rename = "_version")]
    pub version: SperrCodecVersion,
}

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
/// SPERR compression mode
#[serde(tag = "mode")]
pub enum SperrCompressionMode {
    /// Fixed bit-per-pixel rate
    #[serde(rename = "bpp")]
    BitsPerPixel {
        /// positive bits-per-pixel
        bpp: Positive<f64>,
    },
    /// Fixed peak signal-to-noise ratio
    #[serde(rename = "psnr")]
    PeakSignalToNoiseRatio {
        /// positive peak signal-to-noise ratio
        psnr: Positive<f64>,
    },
    /// Fixed point-wise (absolute) error
    #[serde(rename = "pwe")]
    PointwiseError {
        /// positive point-wise (absolute) error
        pwe: Positive<f64>,
    },
}

impl Codec for SperrCodec {
    type Error = SperrCodecError;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        match data {
            AnyCowArray::F32(data) => Ok(AnyArray::U8(
                Array1::from(compress(data, &self.mode)?).into_dyn(),
            )),
            AnyCowArray::F64(data) => Ok(AnyArray::U8(
                Array1::from(compress(data, &self.mode)?).into_dyn(),
            )),
            encoded => Err(SperrCodecError::UnsupportedDtype(encoded.dtype())),
        }
    }

    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
        let AnyCowArray::U8(encoded) = encoded else {
            return Err(SperrCodecError::EncodedDataNotBytes {
                dtype: encoded.dtype(),
            });
        };

        if !matches!(encoded.shape(), [_]) {
            return Err(SperrCodecError::EncodedDataNotOneDimensional {
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

impl StaticCodec for SperrCodec {
    const CODEC_ID: &'static str = "sperr.rs";

    type Config<'de> = Self;

    fn from_config(config: Self::Config<'_>) -> Self {
        config
    }

    fn get_config(&self) -> StaticCodecConfig<'_, Self> {
        StaticCodecConfig::from(self)
    }
}

#[derive(Debug, Error)]
/// Errors that may occur when applying the [`SperrCodec`].
pub enum SperrCodecError {
    /// [`SperrCodec`] does not support the dtype
    #[error("Sperr does not support the dtype {0}")]
    UnsupportedDtype(AnyArrayDType),
    /// [`SperrCodec`] failed to encode the header
    #[error("Sperr failed to encode the header")]
    HeaderEncodeFailed {
        /// Opaque source error
        source: SperrHeaderError,
    },
    /// [`SperrCodec`] failed to encode the data
    #[error("Sperr failed to encode the data")]
    SperrEncodeFailed {
        /// Opaque source error
        source: SperrCodingError,
    },
    /// [`SperrCodec`] failed to encode a slice
    #[error("Sperr failed to encode a slice")]
    SliceEncodeFailed {
        /// Opaque source error
        source: SperrSliceError,
    },
    /// [`SperrCodec`] can only decode one-dimensional byte arrays but received
    /// an array of a different dtype
    #[error(
        "Sperr can only decode one-dimensional byte arrays but received an array of dtype {dtype}"
    )]
    EncodedDataNotBytes {
        /// The unexpected dtype of the encoded array
        dtype: AnyArrayDType,
    },
    /// [`SperrCodec`] can only decode one-dimensional byte arrays but received
    /// an array of a different shape
    #[error(
        "Sperr can only decode one-dimensional byte arrays but received a byte array of shape {shape:?}"
    )]
    EncodedDataNotOneDimensional {
        /// The unexpected shape of the encoded array
        shape: Vec<usize>,
    },
    /// [`SperrCodec`] failed to decode the header
    #[error("Sperr failed to decode the header")]
    HeaderDecodeFailed {
        /// Opaque source error
        source: SperrHeaderError,
    },
    /// [`SperrCodec`] failed to decode a slice
    #[error("Sperr failed to decode a slice")]
    SliceDecodeFailed {
        /// Opaque source error
        source: SperrSliceError,
    },
    /// [`SperrCodec`] failed to decode from an excessive number of slices
    #[error("Sperr failed to decode from an excessive number of slices")]
    DecodeTooManySlices,
    /// [`SperrCodec`] failed to decode the data
    #[error("Sperr failed to decode the data")]
    SperrDecodeFailed {
        /// Opaque source error
        source: SperrCodingError,
    },
    /// [`SperrCodec`] decoded into an invalid shape not matching the data size
    #[error("Sperr decoded into an invalid shape not matching the data size")]
    DecodeInvalidShape {
        /// The source of the error
        source: ShapeError,
    },
    /// [`SperrCodec`] cannot decode into the provided array
    #[error("Sperr cannot decode into the provided array")]
    MismatchedDecodeIntoArray {
        /// The source of the error
        #[from]
        source: AnyArrayAssignError,
    },
}

#[derive(Debug, Error)]
#[error(transparent)]
/// Opaque error for when encoding or decoding the header fails
pub struct SperrHeaderError(postcard::Error);

#[derive(Debug, Error)]
#[error(transparent)]
/// Opaque error for when encoding or decoding a slice fails
pub struct SperrSliceError(postcard::Error);

#[derive(Debug, Error)]
#[error(transparent)]
/// Opaque error for when encoding or decoding with SPERR fails
pub struct SperrCodingError(sperr::Error);

/// Compress the `data` array using SPERR with the provided `mode`.
///
/// # Errors
///
/// Errors with
/// - [`SperrCodecError::HeaderEncodeFailed`] if encoding the header failed
/// - [`SperrCodecError::SperrEncodeFailed`] if encoding with SPERR failed
/// - [`SperrCodecError::SliceEncodeFailed`] if encoding a slice failed
#[allow(clippy::missing_panics_doc)]
pub fn compress<T: SperrElement, S: Data<Elem = T>, D: Dimension>(
    data: ArrayBase<S, D>,
    mode: &SperrCompressionMode,
) -> Result<Vec<u8>, SperrCodecError> {
    let mut encoded = postcard::to_extend(
        &CompressionHeader {
            dtype: T::DTYPE,
            shape: Cow::Borrowed(data.shape()),
            version: StaticCodecVersion,
        },
        Vec::new(),
    )
    .map_err(|err| SperrCodecError::HeaderEncodeFailed {
        source: SperrHeaderError(err),
    })?;

    // SPERR cannot handle zero-length dimensions
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
        [width] => (width, 1, 1),
        [] => (1, 1, 1),
    };

    for mut slice in data.into_dyn().exact_chunks(chunk_size.as_slice()) {
        while slice.ndim() < 3 {
            slice = slice.insert_axis(Axis(0));
        }
        #[allow(clippy::unwrap_used)]
        // slice must now have at least three axes, and all but the last three
        //  must be of size 1
        let slice = slice.into_shape_with_order((depth, height, width)).unwrap();

        let encoded_slice = sperr::compress_3d(
            slice,
            match mode {
                SperrCompressionMode::BitsPerPixel { bpp } => {
                    sperr::CompressionMode::BitsPerPixel { bpp: bpp.0 }
                }
                SperrCompressionMode::PeakSignalToNoiseRatio { psnr } => {
                    sperr::CompressionMode::PeakSignalToNoiseRatio { psnr: psnr.0 }
                }
                SperrCompressionMode::PointwiseError { pwe } => {
                    sperr::CompressionMode::PointwiseError { pwe: pwe.0 }
                }
            },
            (256, 256, 256),
        )
        .map_err(|err| SperrCodecError::SperrEncodeFailed {
            source: SperrCodingError(err),
        })?;

        encoded = postcard::to_extend(encoded_slice.as_slice(), encoded).map_err(|err| {
            SperrCodecError::SliceEncodeFailed {
                source: SperrSliceError(err),
            }
        })?;
    }

    Ok(encoded)
}

/// Decompress the `encoded` data into an array using SPERR.
///
/// # Errors
///
/// Errors with
/// - [`SperrCodecError::HeaderDecodeFailed`] if decoding the header failed
/// - [`SperrCodecError::SliceDecodeFailed`] if decoding a slice failed
/// - [`SperrCodecError::SperrDecodeFailed`] if decoding with SPERR failed
/// - [`SperrCodecError::DecodeInvalidShape`] if the encoded data decodes to
///   an unexpected shape
/// - [`SperrCodecError::DecodeTooManySlices`] if the encoded data contains
///   too many slices
pub fn decompress(encoded: &[u8]) -> Result<AnyArray, SperrCodecError> {
    fn decompress_typed<T: SperrElement>(
        mut encoded: &[u8],
        shape: &[usize],
    ) -> Result<Array<T, IxDyn>, SperrCodecError> {
        let mut decoded = Array::<T, _>::zeros(shape);

        let mut chunk_size = Vec::from(shape);
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
                    SperrCodecError::SliceDecodeFailed {
                        source: SperrSliceError(err),
                    }
                })?;
            encoded = rest;

            while slice.ndim() < 3 {
                slice = slice.insert_axis(Axis(0));
            }
            #[allow(clippy::unwrap_used)]
            // slice must now have at least three axes, and all but the last
            //  three must be of size 1
            let slice = slice.into_shape_with_order((depth, height, width)).unwrap();

            sperr::decompress_into_3d(&encoded_slice, slice).map_err(|err| {
                SperrCodecError::SperrDecodeFailed {
                    source: SperrCodingError(err),
                }
            })?;
        }

        if !encoded.is_empty() {
            return Err(SperrCodecError::DecodeTooManySlices);
        }

        Ok(decoded)
    }

    let (header, encoded) =
        postcard::take_from_bytes::<CompressionHeader>(encoded).map_err(|err| {
            SperrCodecError::HeaderDecodeFailed {
                source: SperrHeaderError(err),
            }
        })?;

    // Return empty data for zero-size arrays
    if header.shape.iter().copied().product::<usize>() == 0 {
        return match header.dtype {
            SperrDType::F32 => Ok(AnyArray::F32(Array::zeros(&*header.shape))),
            SperrDType::F64 => Ok(AnyArray::F64(Array::zeros(&*header.shape))),
        };
    }

    match header.dtype {
        SperrDType::F32 => Ok(AnyArray::F32(decompress_typed(encoded, &header.shape)?)),
        SperrDType::F64 => Ok(AnyArray::F64(decompress_typed(encoded, &header.shape)?)),
    }
}

/// Array element types which can be compressed with SPERR.
pub trait SperrElement: sperr::Element + Zero {
    /// The dtype representation of the type
    const DTYPE: SperrDType;
}

impl SperrElement for f32 {
    const DTYPE: SperrDType = SperrDType::F32;
}
impl SperrElement for f64 {
    const DTYPE: SperrDType = SperrDType::F64;
}

#[expect(clippy::derive_partial_eq_without_eq)] // floats are not Eq
#[derive(Copy, Clone, PartialEq, PartialOrd, Hash)]
/// Positive floating point number
pub struct Positive<T: Float>(T);

impl<T: Float> Positive<T> {
    #[must_use]
    /// Get the positive floating point value
    pub const fn get(self) -> T {
        self.0
    }
}

impl Serialize for Positive<f64> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_f64(self.0)
    }
}

impl<'de> Deserialize<'de> for Positive<f64> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let x = f64::deserialize(deserializer)?;

        if x > 0.0 {
            Ok(Self(x))
        } else {
            Err(serde::de::Error::invalid_value(
                serde::de::Unexpected::Float(x),
                &"a positive value",
            ))
        }
    }
}

impl JsonSchema for Positive<f64> {
    fn schema_name() -> Cow<'static, str> {
        Cow::Borrowed("PositiveF64")
    }

    fn schema_id() -> Cow<'static, str> {
        Cow::Borrowed(concat!(module_path!(), "::", "Positive<f64>"))
    }

    fn json_schema(_gen: &mut SchemaGenerator) -> Schema {
        json_schema!({
            "type": "number",
            "exclusiveMinimum": 0.0
        })
    }
}

#[derive(Serialize, Deserialize)]
struct CompressionHeader<'a> {
    dtype: SperrDType,
    #[serde(borrow)]
    shape: Cow<'a, [usize]>,
    version: SperrCodecVersion,
}

/// Dtypes that SPERR can compress and decompress
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
#[expect(missing_docs)]
pub enum SperrDType {
    #[serde(rename = "f32", alias = "float32")]
    F32,
    #[serde(rename = "f64", alias = "float64")]
    F64,
}

impl fmt::Display for SperrDType {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.write_str(match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use ndarray::{Ix0, Ix1, Ix2, Ix3, Ix4};

    use super::*;

    #[test]
    fn zero_length() {
        let encoded = compress(
            Array::<f32, _>::from_shape_vec([3, 0], vec![]).unwrap(),
            &SperrCompressionMode::PeakSignalToNoiseRatio {
                psnr: Positive(42.0),
            },
        )
        .unwrap();
        let decoded = decompress(&encoded).unwrap();

        assert_eq!(decoded.dtype(), AnyArrayDType::F32);
        assert!(decoded.is_empty());
        assert_eq!(decoded.shape(), &[3, 0]);
    }

    #[test]
    fn small_2d() {
        let encoded = compress(
            Array::<f32, _>::from_shape_vec([1, 1], vec![42.0]).unwrap(),
            &SperrCompressionMode::PeakSignalToNoiseRatio {
                psnr: Positive(42.0),
            },
        )
        .unwrap();
        let decoded = decompress(&encoded).unwrap();

        assert_eq!(decoded.dtype(), AnyArrayDType::F32);
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded.shape(), &[1, 1]);
    }

    #[test]
    fn large_3d() {
        let encoded = compress(
            Array::<f64, _>::zeros((64, 64, 64)),
            &SperrCompressionMode::PeakSignalToNoiseRatio {
                psnr: Positive(42.0),
            },
        )
        .unwrap();
        let decoded = decompress(&encoded).unwrap();

        assert_eq!(decoded.dtype(), AnyArrayDType::F64);
        assert_eq!(decoded.len(), 64 * 64 * 64);
        assert_eq!(decoded.shape(), &[64, 64, 64]);
    }

    #[test]
    fn all_modes() {
        for mode in [
            SperrCompressionMode::BitsPerPixel { bpp: Positive(1.0) },
            SperrCompressionMode::PeakSignalToNoiseRatio {
                psnr: Positive(42.0),
            },
            SperrCompressionMode::PointwiseError { pwe: Positive(0.1) },
        ] {
            let encoded = compress(Array::<f64, _>::zeros((64, 64, 64)), &mode).unwrap();
            let decoded = decompress(&encoded).unwrap();

            assert_eq!(decoded.dtype(), AnyArrayDType::F64);
            assert_eq!(decoded.len(), 64 * 64 * 64);
            assert_eq!(decoded.shape(), &[64, 64, 64]);
        }
    }

    #[test]
    fn many_dimensions() {
        for data in [
            Array::<f32, Ix0>::from_shape_vec([], vec![42.0])
                .unwrap()
                .into_dyn(),
            Array::<f32, Ix1>::from_shape_vec([2], vec![1.0, 2.0])
                .unwrap()
                .into_dyn(),
            Array::<f32, Ix2>::from_shape_vec([2, 2], vec![1.0, 2.0, 3.0, 4.0])
                .unwrap()
                .into_dyn(),
            Array::<f32, Ix3>::from_shape_vec(
                [2, 2, 2],
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            )
            .unwrap()
            .into_dyn(),
            Array::<f32, Ix4>::from_shape_vec(
                [2, 2, 2, 2],
                vec![
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                    15.0, 16.0,
                ],
            )
            .unwrap()
            .into_dyn(),
        ] {
            let encoded = compress(
                data.view(),
                &SperrCompressionMode::PointwiseError {
                    pwe: Positive(f64::EPSILON),
                },
            )
            .unwrap();
            let decoded = decompress(&encoded).unwrap();

            assert_eq!(decoded, AnyArray::F32(data));
        }
    }
}
