//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.87.0-blue
//! [repo]: https://github.com/juntyr/numcodecs-rs
//!
//! [Latest Version]: https://img.shields.io/crates/v/numcodecs-qpet-sperr
//! [crates.io]: https://crates.io/crates/numcodecs-qpet-sperr
//!
//! [Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs-qpet-sperr
//! [docs.rs]: https://docs.rs/numcodecs-qpet-sperr/
//!
//! [Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/numcodecs-rs/numcodecs_qpet_sperr
//!
//! QPET-SPERR codec implementation for the [`numcodecs`] API.

#![allow(clippy::multiple_crate_versions)] // embedded-io

// Only included to explicitly enable the `no_wasm_shim` feature for
// qpet-sperr-sys/zstd-sys
use ::zstd_sys as _;

#[cfg(test)]
use ::serde_json as _;

use std::{
    borrow::Cow,
    fmt,
    num::{NonZeroU16, NonZeroUsize},
};

use ndarray::{Array, Array1, ArrayBase, Axis, Data, Dimension, IxDyn, ShapeError};
use num_traits::{Float, identities::Zero};
use numcodecs::{
    AnyArray, AnyArrayAssignError, AnyArrayDType, AnyArrayView, AnyArrayViewMut, AnyCowArray,
    Codec, StaticCodec, StaticCodecConfig, StaticCodecVersion,
};
use schemars::{JsonSchema, Schema, SchemaGenerator, json_schema};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

type QpetSperrCodecVersion = StaticCodecVersion<0, 1, 0>;

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
// serde cannot deny unknown fields because of the flatten
#[schemars(deny_unknown_fields)]
/// Codec providing compression using QPET-SPERR.
///
/// Arrays that are higher-dimensional than 3D are encoded by compressing each
/// 3D slice with QPET-SPERR independently. Specifically, the array's shape is
/// interpreted as `[.., depth, height, width]`. If you want to compress 3D
/// slices along three different axes, you can swizzle the array axes
/// beforehand.
pub struct QpetSperrCodec {
    /// QPET-SPERR compression mode
    #[serde(flatten)]
    pub mode: QpetSperrCompressionMode,
    /// The codec's encoding format version. Do not provide this parameter explicitly.
    #[serde(default, rename = "_version")]
    pub version: QpetSperrCodecVersion,
}

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
/// QPET-SPERR compression mode
#[serde(tag = "mode")]
pub enum QpetSperrCompressionMode {
    /// Symbolic Quantity of Interest
    #[serde(rename = "qoi-symbolic")]
    SymbolicQuantityOfInterest {
        /// quantity of interest expression
        qoi: String,
        /// block size over which the quantity of interest errors are averaged,
        /// 1 for pointwise
        #[serde(default = "default_qoi_block_size")]
        qoi_block_size: NonZeroU16,
        /// positive (pointwise) absolute error bound over the quantity of
        /// interest
        qoi_pwe: Positive<f64>,
        /// 3D size of the chunks (z, y, x) that SPERR uses internally
        #[serde(default = "default_sperr_chunks")]
        sperr_chunks: (NonZeroUsize, NonZeroUsize, NonZeroUsize),
        /// optional positive pointwise absolute error bound over the data
        #[serde(default)]
        data_pwe: Option<Positive<f64>>,
        /// positive quantity of interest k parameter (3.0 is a good default)
        #[serde(default = "default_qoi_k")]
        qoi_k: Positive<f64>,
        /// high precision mode for SPERR, useful for small error bounds
        #[serde(default)]
        high_prec: bool,
    },
}

const fn default_qoi_block_size() -> NonZeroU16 {
    const NON_ZERO_ONE: NonZeroU16 = NonZeroU16::MIN;
    // 1: pointwise
    NON_ZERO_ONE
}

const fn default_sperr_chunks() -> (NonZeroUsize, NonZeroUsize, NonZeroUsize) {
    const NON_ZERO_256: NonZeroUsize = NonZeroUsize::MIN.saturating_add(255);
    (NON_ZERO_256, NON_ZERO_256, NON_ZERO_256)
}

const fn default_qoi_k() -> Positive<f64> {
    // c=3.0, suggested default
    Positive(3.0)
}

impl Codec for QpetSperrCodec {
    type Error = QpetSperrCodecError;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        match data {
            AnyCowArray::F32(data) => Ok(AnyArray::U8(
                Array1::from(compress(data, &self.mode)?).into_dyn(),
            )),
            AnyCowArray::F64(data) => Ok(AnyArray::U8(
                Array1::from(compress(data, &self.mode)?).into_dyn(),
            )),
            encoded => Err(QpetSperrCodecError::UnsupportedDtype(encoded.dtype())),
        }
    }

    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
        let AnyCowArray::U8(encoded) = encoded else {
            return Err(QpetSperrCodecError::EncodedDataNotBytes {
                dtype: encoded.dtype(),
            });
        };

        if !matches!(encoded.shape(), [_]) {
            return Err(QpetSperrCodecError::EncodedDataNotOneDimensional {
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

impl StaticCodec for QpetSperrCodec {
    const CODEC_ID: &'static str = "qpet-sperr.rs";

    type Config<'de> = Self;

    fn from_config(config: Self::Config<'_>) -> Self {
        config
    }

    fn get_config(&self) -> StaticCodecConfig<'_, Self> {
        StaticCodecConfig::from(self)
    }
}

#[derive(Debug, Error)]
/// Errors that may occur when applying the [`QpetSperrCodec`].
pub enum QpetSperrCodecError {
    /// [`QpetSperrCodec`] does not support the dtype
    #[error("QpetSperr does not support the dtype {0}")]
    UnsupportedDtype(AnyArrayDType),
    /// [`QpetSperrCodec`] failed to encode the header
    #[error("QpetSperr failed to encode the header")]
    HeaderEncodeFailed {
        /// Opaque source error
        source: QpetSperrHeaderError,
    },
    /// [`QpetSperrCodec`] failed to encode the data
    #[error("QpetSperr failed to encode the data")]
    QpetSperrEncodeFailed {
        /// Opaque source error
        source: QpetSperrCodingError,
    },
    /// [`QpetSperrCodec`] failed to encode a slice
    #[error("QpetSperr failed to encode a slice")]
    SliceEncodeFailed {
        /// Opaque source error
        source: QpetSperrSliceError,
    },
    /// [`QpetSperrCodec`] can only decode one-dimensional byte arrays but
    /// received an array of a different dtype
    #[error(
        "QpetSperr can only decode one-dimensional byte arrays but received an array of dtype {dtype}"
    )]
    EncodedDataNotBytes {
        /// The unexpected dtype of the encoded array
        dtype: AnyArrayDType,
    },
    /// [`QpetSperrCodec`] can only decode one-dimensional byte arrays but
    /// received an array of a different shape
    #[error(
        "QpetSperr can only decode one-dimensional byte arrays but received a byte array of shape {shape:?}"
    )]
    EncodedDataNotOneDimensional {
        /// The unexpected shape of the encoded array
        shape: Vec<usize>,
    },
    /// [`QpetSperrCodec`] failed to decode the header
    #[error("QpetSperr failed to decode the header")]
    HeaderDecodeFailed {
        /// Opaque source error
        source: QpetSperrHeaderError,
    },
    /// [`QpetSperrCodec`] failed to decode a slice
    #[error("QpetSperr failed to decode a slice")]
    SliceDecodeFailed {
        /// Opaque source error
        source: QpetSperrSliceError,
    },
    /// [`QpetSperrCodec`] failed to decode from an excessive number of slices
    #[error("QpetSperr failed to decode from an excessive number of slices")]
    DecodeTooManySlices,
    /// [`QpetSperrCodec`] failed to decode the data
    #[error("QpetSperr failed to decode the data")]
    SperrDecodeFailed {
        /// Opaque source error
        source: QpetSperrCodingError,
    },
    /// [`QpetSperrCodec`] decoded into an invalid shape not matching the data size
    #[error("QpetSperr decoded into an invalid shape not matching the data size")]
    DecodeInvalidShape {
        /// The source of the error
        source: ShapeError,
    },
    /// [`QpetSperrCodec`] cannot decode into the provided array
    #[error("QpetSperr cannot decode into the provided array")]
    MismatchedDecodeIntoArray {
        /// The source of the error
        #[from]
        source: AnyArrayAssignError,
    },
}

#[derive(Debug, Error)]
#[error(transparent)]
/// Opaque error for when encoding or decoding the header fails
pub struct QpetSperrHeaderError(postcard::Error);

#[derive(Debug, Error)]
#[error(transparent)]
/// Opaque error for when encoding or decoding a slice fails
pub struct QpetSperrSliceError(postcard::Error);

#[derive(Debug, Error)]
#[error(transparent)]
/// Opaque error for when encoding or decoding with SPERR fails
pub struct QpetSperrCodingError(qpet_sperr::Error);

/// Compress the `data` array using QPET-SPERR with the provided `mode`.
///
/// The compressed data can be decompressed using SPERR or QPET-SPERR.
///
/// # Errors
///
/// Errors with
/// - [`QpetSperrCodecError::HeaderEncodeFailed`] if encoding the header failed
/// - [`QpetSperrCodecError::QpetSperrEncodeFailed`] if encoding with
///   QPET-SPERR failed
/// - [`QpetSperrCodecError::SliceEncodeFailed`] if encoding a slice failed
#[allow(clippy::missing_panics_doc)]
pub fn compress<T: QpetSperrElement, S: Data<Elem = T>, D: Dimension>(
    data: ArrayBase<S, D>,
    mode: &QpetSperrCompressionMode,
) -> Result<Vec<u8>, QpetSperrCodecError> {
    let mut encoded = postcard::to_extend(
        &CompressionHeader {
            dtype: T::DTYPE,
            shape: Cow::Borrowed(data.shape()),
            version: StaticCodecVersion,
        },
        Vec::new(),
    )
    .map_err(|err| QpetSperrCodecError::HeaderEncodeFailed {
        source: QpetSperrHeaderError(err),
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

        let QpetSperrCompressionMode::SymbolicQuantityOfInterest {
            qoi,
            qoi_block_size,
            qoi_pwe,
            sperr_chunks,
            data_pwe,
            qoi_k,
            high_prec,
        } = mode;

        let encoded_slice = qpet_sperr::compress_3d(
            slice,
            qpet_sperr::CompressionMode::SymbolicQuantityOfInterest {
                qoi: qoi.as_str(),
                qoi_block_size: *qoi_block_size,
                qoi_pwe: qoi_pwe.0,
                data_pwe: data_pwe.map(|data_pwe| data_pwe.0),
                qoi_k: qoi_k.0,
                high_prec: *high_prec,
            },
            (
                sperr_chunks.0.get(),
                sperr_chunks.1.get(),
                sperr_chunks.2.get(),
            ),
        )
        .map_err(|err| QpetSperrCodecError::QpetSperrEncodeFailed {
            source: QpetSperrCodingError(err),
        })?;

        encoded = postcard::to_extend(encoded_slice.as_slice(), encoded).map_err(|err| {
            QpetSperrCodecError::SliceEncodeFailed {
                source: QpetSperrSliceError(err),
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
/// - [`QpetSperrCodecError::HeaderDecodeFailed`] if decoding the header failed
/// - [`QpetSperrCodecError::SliceDecodeFailed`] if decoding a slice failed
/// - [`QpetSperrCodecError::SperrDecodeFailed`] if decoding with SPERR failed
/// - [`QpetSperrCodecError::DecodeInvalidShape`] if the encoded data decodes
///   to an unexpected shape
/// - [`QpetSperrCodecError::DecodeTooManySlices`] if the encoded data contains
///   too many slices
pub fn decompress(encoded: &[u8]) -> Result<AnyArray, QpetSperrCodecError> {
    fn decompress_typed<T: QpetSperrElement>(
        mut encoded: &[u8],
        shape: &[usize],
    ) -> Result<Array<T, IxDyn>, QpetSperrCodecError> {
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
                    QpetSperrCodecError::SliceDecodeFailed {
                        source: QpetSperrSliceError(err),
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

            qpet_sperr::decompress_into_3d(&encoded_slice, slice).map_err(|err| {
                QpetSperrCodecError::SperrDecodeFailed {
                    source: QpetSperrCodingError(err),
                }
            })?;
        }

        if !encoded.is_empty() {
            return Err(QpetSperrCodecError::DecodeTooManySlices);
        }

        Ok(decoded)
    }

    let (header, encoded) =
        postcard::take_from_bytes::<CompressionHeader>(encoded).map_err(|err| {
            QpetSperrCodecError::HeaderDecodeFailed {
                source: QpetSperrHeaderError(err),
            }
        })?;

    // Return empty data for zero-size arrays
    if header.shape.iter().copied().product::<usize>() == 0 {
        return match header.dtype {
            QpetSperrDType::F32 => Ok(AnyArray::F32(Array::zeros(&*header.shape))),
            QpetSperrDType::F64 => Ok(AnyArray::F64(Array::zeros(&*header.shape))),
        };
    }

    match header.dtype {
        QpetSperrDType::F32 => Ok(AnyArray::F32(decompress_typed(encoded, &header.shape)?)),
        QpetSperrDType::F64 => Ok(AnyArray::F64(decompress_typed(encoded, &header.shape)?)),
    }
}

/// Array element types which can be compressed with QPET-SPERR.
pub trait QpetSperrElement: qpet_sperr::Element + Zero {
    /// The dtype representation of the type
    const DTYPE: QpetSperrDType;
}

impl QpetSperrElement for f32 {
    const DTYPE: QpetSperrDType = QpetSperrDType::F32;
}
impl QpetSperrElement for f64 {
    const DTYPE: QpetSperrDType = QpetSperrDType::F64;
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
    dtype: QpetSperrDType,
    #[serde(borrow)]
    shape: Cow<'a, [usize]>,
    version: QpetSperrCodecVersion,
}

/// Dtypes that QPET-SPERR can compress and decompress
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
#[expect(missing_docs)]
pub enum QpetSperrDType {
    #[serde(rename = "f32", alias = "float32")]
    F32,
    #[serde(rename = "f64", alias = "float64")]
    F64,
}

impl fmt::Display for QpetSperrDType {
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
    use std::f64;

    use ndarray::{Ix0, Ix1, Ix2, Ix3, Ix4};

    use super::*;

    #[test]
    fn zero_length() {
        let encoded = compress(
            Array::<f32, _>::from_shape_vec([3, 0], vec![]).unwrap(),
            &QpetSperrCompressionMode::SymbolicQuantityOfInterest {
                qoi: String::from("x"),
                qoi_block_size: default_qoi_block_size(),
                qoi_pwe: Positive(42.0),
                sperr_chunks: default_sperr_chunks(),
                data_pwe: None,
                qoi_k: default_qoi_k(),
                high_prec: false,
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
            &QpetSperrCompressionMode::SymbolicQuantityOfInterest {
                qoi: String::from("x"),
                qoi_block_size: default_qoi_block_size(),
                qoi_pwe: Positive(42.0),
                sperr_chunks: default_sperr_chunks(),
                data_pwe: None,
                qoi_k: default_qoi_k(),
                high_prec: false,
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
            &QpetSperrCompressionMode::SymbolicQuantityOfInterest {
                qoi: String::from("x"),
                qoi_block_size: default_qoi_block_size(),
                qoi_pwe: Positive(42.0),
                sperr_chunks: default_sperr_chunks(),
                data_pwe: None,
                qoi_k: default_qoi_k(),
                high_prec: false,
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
        for mode in [QpetSperrCompressionMode::SymbolicQuantityOfInterest {
            qoi: String::from("x^2"),
            qoi_block_size: default_qoi_block_size(),
            qoi_pwe: Positive(0.1),
            sperr_chunks: default_sperr_chunks(),
            data_pwe: None,
            qoi_k: default_qoi_k(),
            high_prec: false,
        }] {
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
                &QpetSperrCompressionMode::SymbolicQuantityOfInterest {
                    qoi: String::from("x"),
                    qoi_block_size: default_qoi_block_size(),
                    qoi_pwe: Positive(f64::EPSILON),
                    sperr_chunks: default_sperr_chunks(),
                    data_pwe: None,
                    qoi_k: default_qoi_k(),
                    high_prec: false,
                },
            )
            .unwrap();
            let decoded = decompress(&encoded).unwrap();

            assert_eq!(decoded, AnyArray::F32(data));
        }
    }

    #[test]
    fn zero_square_qoi() {
        let encoded = compress(
            Array::<f64, _>::zeros((64, 64, 1)),
            &QpetSperrCompressionMode::SymbolicQuantityOfInterest {
                qoi: String::from("x^2"),
                qoi_block_size: default_qoi_block_size(),
                qoi_pwe: Positive(0.1),
                sperr_chunks: default_sperr_chunks(),
                data_pwe: None,
                qoi_k: default_qoi_k(),
                high_prec: false,
            },
        )
        .unwrap();
        let decoded = decompress(&encoded).unwrap();

        assert_eq!(decoded.dtype(), AnyArrayDType::F64);
        assert_eq!(decoded.len(), 64 * 64 * 1);
        assert_eq!(decoded.shape(), &[64, 64, 1]);
    }
}
