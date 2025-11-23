//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.87.0-blue
//! [repo]: https://github.com/juntyr/numcodecs-rs
//!
//! [Latest Version]: https://img.shields.io/crates/v/numcodecs-qpet-sz
//! [crates.io]: https://crates.io/crates/numcodecs-qpet-sz
//!
//! [Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs-qpet-sz
//! [docs.rs]: https://docs.rs/numcodecs-qpet-sz/
//!
//! [Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/numcodecs-rs/numcodecs_qpet_sz
//!
//! QPET-SZ codec implementation for the [`numcodecs`] API.

#![allow(clippy::multiple_crate_versions)] // embedded-io

// Only included to explicitly enable the `no_wasm_shim` feature for
// qpet-sz-sys/zstd-sys
use ::zstd_sys as _;

#[cfg(test)]
use ::serde_json as _;

use std::{borrow::Cow, fmt, num::NonZeroUsize};

use ndarray::{Array, Array1, ArrayBase, CowArray, Data, Dimension, ShapeError};
use num_traits::{Float, identities::Zero};
use numcodecs::{
    AnyArray, AnyArrayAssignError, AnyArrayDType, AnyArrayView, AnyArrayViewMut, AnyCowArray,
    Codec, StaticCodec, StaticCodecConfig, StaticCodecVersion,
};
use schemars::{JsonSchema, Schema, SchemaGenerator, json_schema};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

type QpetSzCodecVersion = StaticCodecVersion<0, 1, 0>;

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
// serde cannot deny unknown fields because of the flatten
#[schemars(deny_unknown_fields)]
/// Codec providing compression using QPET-SZ.
pub struct QpetSzCodec {
    /// QPET-SZ compression mode
    #[serde(flatten)]
    pub mode: QpetSzCompressionMode,
    /// The codec's encoding format version. Do not provide this parameter explicitly.
    #[serde(default, rename = "_version")]
    pub version: QpetSzCodecVersion,
}

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
/// QPET-SZ compression mode
#[serde(tag = "mode")]
pub enum QpetSzCompressionMode {
    /// Symbolic Quantity of Interest
    #[serde(rename = "qoi-symbolic")]
    SymbolicQuantityOfInterest {
        /// quantity of interest expression
        qoi: String,
        /// side length of the region of the quantity of interest,
        /// 1 for pointwise
        #[serde(default = "default_qoi_region_size")]
        qoi_region_size: NonZeroUsize,
        /// quantity of interest error bound
        #[serde(flatten)]
        qoi_error_bound: QpetSzQoIErrorBound,
        // /// data error bound
        // #[serde(flatten)]
        // data_error_bound: QpetSzDataErrorBound,
        /// positive quantity of interest c parameter (2.0 is a good default)
        #[serde(default = "default_qoi_c")]
        qoi_c: Positive<f64>,
    },
}

// /// QPET-SZ data error bound
// #[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
// #[serde(tag = "data_eb_mode")]
// #[serde(deny_unknown_fields)]
// pub enum QpetSzDataErrorBound {
//     /// Errors are bounded by *both* the absolute and range-relative error,
//     /// i.e. by whichever bound is stricter
//     #[serde(rename = "abs-and-rel")]
//     AbsoluteAndRelative {
//         /// Absolute error bound
//         #[serde(rename = "data_eb_abs")]
//         abs: Positive<f64>,
//         /// Relative error bound
//         #[serde(rename = "data_eb_rel")]
//         rel: Positive<f64>,
//     },
//     /// Errors are bounded by *either* the absolute or range-relative error,
//     /// i.e. by whichever bound is weaker
//     #[serde(rename = "abs-or-rel")]
//     AbsoluteOrRelative {
//         /// Absolute error bound
//         #[serde(rename = "data_eb_abs")]
//         abs: Positive<f64>,
//         /// Relative error bound
//         #[serde(rename = "data_eb_rel")]
//         rel: Positive<f64>,
//     },
//     /// Absolute error bound
//     #[serde(rename = "abs")]
//     Absolute {
//         /// Absolute error bound
//         #[serde(rename = "data_eb_abs")]
//         abs: Positive<f64>,
//     },
//     /// Range-relative error bound
//     #[serde(rename = "rel")]
//     Relative {
//         /// Range-relative error bound
//         #[serde(rename = "data_eb_rel")]
//         rel: Positive<f64>,
//     },
//     /// Peak signal to noise ratio error bound
//     #[serde(rename = "psnr")]
//     PS2NR {
//         /// Peak signal to noise ratio error bound
//         #[serde(rename = "eb_psnr")]
//         psnr: Positive<f64>,
//     },
//     /// Peak L2 norm error bound
//     #[serde(rename = "l2")]
//     L2Norm {
//         /// Peak L2 norm error bound
//         #[serde(rename = "eb_l2")]
//         l2: Positive<f64>,
//     },
// }

/// QPET-SZ QoI error bound
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "qoi_eb_mode")]
#[serde(deny_unknown_fields)]
pub enum QpetSzQoIErrorBound {
    /// Absolute error bound
    #[serde(rename = "abs")]
    Absolute {
        /// Absolute error bound
        #[serde(rename = "qoi_eb_abs")]
        abs: Positive<f64>,
    },
    /// Range-relative error bound
    #[serde(rename = "rel")]
    Relative {
        /// Range-relative error bound
        #[serde(rename = "qoi_eb_rel")]
        rel: Positive<f64>,
    },
}

const fn default_qoi_region_size() -> NonZeroUsize {
    const NON_ZERO_ONE: NonZeroUsize = NonZeroUsize::MIN;
    // 1: pointwise
    NON_ZERO_ONE
}

const fn default_qoi_c() -> Positive<f64> {
    // c=2.0, suggested default
    Positive(2.0)
}

impl Codec for QpetSzCodec {
    type Error = QpetSzCodecError;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        match data {
            AnyCowArray::F32(data) => Ok(AnyArray::U8(
                Array1::from(compress(data, &self.mode)?).into_dyn(),
            )),
            AnyCowArray::F64(data) => Ok(AnyArray::U8(
                Array1::from(compress(data, &self.mode)?).into_dyn(),
            )),
            encoded => Err(QpetSzCodecError::UnsupportedDtype(encoded.dtype())),
        }
    }

    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
        let AnyCowArray::U8(encoded) = encoded else {
            return Err(QpetSzCodecError::EncodedDataNotBytes {
                dtype: encoded.dtype(),
            });
        };

        if !matches!(encoded.shape(), [_]) {
            return Err(QpetSzCodecError::EncodedDataNotOneDimensional {
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

impl StaticCodec for QpetSzCodec {
    const CODEC_ID: &'static str = "qpet-sz.rs";

    type Config<'de> = Self;

    fn from_config(config: Self::Config<'_>) -> Self {
        config
    }

    fn get_config(&self) -> StaticCodecConfig<'_, Self> {
        StaticCodecConfig::from(self)
    }
}

#[derive(Debug, Error)]
/// Errors that may occur when applying the [`QpetSzCodec`].
pub enum QpetSzCodecError {
    /// [`QpetSzCodec`] does not support the dtype
    #[error("QpetSz does not support the dtype {0}")]
    UnsupportedDtype(AnyArrayDType),
    /// [`QpetSzCodec`] failed to encode the header
    #[error("QpetSz failed to encode the header")]
    HeaderEncodeFailed {
        /// Opaque source error
        source: QpetSzHeaderError,
    },
    /// [`QpetSzCodec`] can only decode one-dimensional byte arrays but
    /// received an array of a different dtype
    #[error(
        "QpetSz can only decode one-dimensional byte arrays but received an array of dtype {dtype}"
    )]
    EncodedDataNotBytes {
        /// The unexpected dtype of the encoded array
        dtype: AnyArrayDType,
    },
    /// [`QpetSzCodec`] can only decode one-dimensional byte arrays but
    /// received an array of a different shape
    #[error(
        "QpetSz can only decode one-dimensional byte arrays but received a byte array of shape {shape:?}"
    )]
    EncodedDataNotOneDimensional {
        /// The unexpected shape of the encoded array
        shape: Vec<usize>,
    },
    /// [`QpetSzCodec`] failed to decode the header
    #[error("QpetSz failed to decode the header")]
    HeaderDecodeFailed {
        /// Opaque source error
        source: QpetSzHeaderError,
    },
    /// [`QpetSzCodec`] decoded an invalid array shape header which does not
    /// fit the decoded data
    #[error("QpetSz decoded an invalid array shape header which does not fit the decoded data")]
    DecodeInvalidShapeHeader {
        /// Source error
        #[from]
        source: ShapeError,
    },
    /// [`QpetSzCodec`] cannot decode into the provided array
    #[error("QpetSz cannot decode into the provided array")]
    MismatchedDecodeIntoArray {
        /// The source of the error
        #[from]
        source: AnyArrayAssignError,
    },
}

#[derive(Debug, Error)]
#[error(transparent)]
/// Opaque error for when encoding or decoding the header fails
pub struct QpetSzHeaderError(postcard::Error);

/// Compress the `data` array using QPET-SZ with the provided `mode`.
///
/// # Errors
///
/// Errors with
/// - [`QpetSzCodecError::HeaderEncodeFailed`] if encoding the header failed
#[allow(clippy::missing_panics_doc)]
pub fn compress<T: QpetSzElement, S: Data<Elem = T>, D: Dimension>(
    data: ArrayBase<S, D>,
    mode: &QpetSzCompressionMode,
) -> Result<Vec<u8>, QpetSzCodecError> {
    let mut encoded_bytes = postcard::to_extend(
        &CompressionHeader {
            dtype: T::DTYPE,
            shape: Cow::Borrowed(data.shape()),
            version: StaticCodecVersion,
        },
        Vec::new(),
    )
    .map_err(|err| QpetSzCodecError::HeaderEncodeFailed {
        source: QpetSzHeaderError(err),
    })?;

    // QPET-SZ cannot handle zero-length dimensions
    if data.is_empty() {
        return Ok(encoded_bytes);
    }

    // QPET-SZ ignores dimensions of length 1
    // Since they carry no information for QPET-SZ and we already encode them
    //  in our custom header, we just skip them here
    let data = data.into_dyn().squeeze();

    // QPET-SZ does not support 0-dimensional (scalar) arrays, so force them to
    //  be one dimensional
    let data = if data.ndim() == 0 {
        data.flatten().into_dyn()
    } else {
        CowArray::from(&data)
    };

    // configure the compression mode
    let QpetSzCompressionMode::SymbolicQuantityOfInterest {
        qoi,
        qoi_region_size,
        qoi_error_bound,
        // data_error_bound,
        qoi_c,
    } = mode;

    let qoi_error_bound = match qoi_error_bound {
        QpetSzQoIErrorBound::Absolute { abs } => qpet_sz::QoIErrorBound::Absolute(abs.0),
        QpetSzQoIErrorBound::Relative { rel } => qpet_sz::QoIErrorBound::Relative(rel.0),
    };

    // let data_error_bound = match data_error_bound {
    //     QpetSzDataErrorBound::AbsoluteAndRelative { abs, rel } => {
    //         qpet_sz::DataErrorBound::AbsoluteAndRelative {
    //             absolute_bound: abs.0,
    //             relative_bound: rel.0,
    //         }
    //     }
    //     QpetSzDataErrorBound::AbsoluteOrRelative { abs, rel } => {
    //         qpet_sz::DataErrorBound::AbsoluteOrRelative {
    //             absolute_bound: abs.0,
    //             relative_bound: rel.0,
    //         }
    //     }
    //     QpetSzDataErrorBound::Absolute { abs } => qpet_sz::DataErrorBound::Absolute(abs.0),
    //     QpetSzDataErrorBound::Relative { rel } => qpet_sz::DataErrorBound::Relative(rel.0),
    //     QpetSzDataErrorBound::PS2NR { psnr } => qpet_sz::DataErrorBound::PSNR(psnr.0),
    //     QpetSzDataErrorBound::L2Norm { l2 } => qpet_sz::DataErrorBound::L2Norm(l2.0),
    // };
    let data_error_bound = qpet_sz::DataErrorBound::Absolute(f64::MAX);

    let config = qpet_sz::Config::new(qpet_sz::CompressionMode::SymbolicQuantityOfInterest {
        qoi: qoi.as_str(),
        qoi_region_size: *qoi_region_size,
        qoi_error_bound,
        data_error_bound,
        qoi_c: qoi_c.0,
    });

    // TODO: avoid extra allocation here
    let compressed = qpet_sz::compress(data.view(), &config);
    encoded_bytes.extend_from_slice(&compressed);

    Ok(encoded_bytes)
}

/// Decompress the `encoded` data into an array using QPET-SZ.
///
/// # Errors
///
/// Errors with
/// - [`QpetSzCodecError::HeaderDecodeFailed`] if decoding the header failed
pub fn decompress(encoded: &[u8]) -> Result<AnyArray, QpetSzCodecError> {
    let (header, data) =
        postcard::take_from_bytes::<CompressionHeader>(encoded).map_err(|err| {
            QpetSzCodecError::HeaderDecodeFailed {
                source: QpetSzHeaderError(err),
            }
        })?;

    let decoded = if header.shape.iter().copied().any(|s| s == 0) {
        match header.dtype {
            QpetSzDType::F32 => {
                AnyArray::F32(Array::from_shape_vec(&*header.shape, Vec::new())?.into_dyn())
            }
            QpetSzDType::F64 => {
                AnyArray::F64(Array::from_shape_vec(&*header.shape, Vec::new())?.into_dyn())
            }
        }
    } else {
        // TODO: avoid extra allocation here
        match header.dtype {
            QpetSzDType::F32 => {
                // FIXME: remove debug print
                let dc = qpet_sz::decompress(data);
                eprintln!("{:?} {:?}", dc.shape(), &*header.shape);
                AnyArray::F32(dc.into_shape_clone(&*header.shape)?)
            }
            QpetSzDType::F64 => {
                // FIXME: remove debug print
                let dc = qpet_sz::decompress(data);
                eprintln!("{:?} {:?}", dc.shape(), &*header.shape);
                AnyArray::F64(dc.into_shape_clone(&*header.shape)?)
            }
        }
    };

    Ok(decoded)
}

/// Array element types which can be compressed with QPET-SZ.
pub trait QpetSzElement: qpet_sz::Element + Zero {
    /// The dtype representation of the type
    const DTYPE: QpetSzDType;
}

impl QpetSzElement for f32 {
    const DTYPE: QpetSzDType = QpetSzDType::F32;
}
impl QpetSzElement for f64 {
    const DTYPE: QpetSzDType = QpetSzDType::F64;
}

#[expect(clippy::derive_partial_eq_without_eq)] // floats are not Eq
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd, Hash)]
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
    dtype: QpetSzDType,
    #[serde(borrow)]
    shape: Cow<'a, [usize]>,
    version: QpetSzCodecVersion,
}

/// Dtypes that QPET-SZ can compress and decompress
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
#[expect(missing_docs)]
pub enum QpetSzDType {
    #[serde(rename = "f32", alias = "float32")]
    F32,
    #[serde(rename = "f64", alias = "float64")]
    F64,
}

impl fmt::Display for QpetSzDType {
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
            &QpetSzCompressionMode::SymbolicQuantityOfInterest {
                qoi: String::from("x"),
                qoi_region_size: default_qoi_region_size(),
                qoi_error_bound: QpetSzQoIErrorBound::Absolute {
                    abs: Positive(42.0),
                },
                // data_error_bound: QpetSzDataErrorBound::Absolute {
                //     abs: Positive(f64::MAX),
                // },
                qoi_c: default_qoi_c(),
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
            &QpetSzCompressionMode::SymbolicQuantityOfInterest {
                qoi: String::from("x"),
                qoi_region_size: default_qoi_region_size(),
                qoi_error_bound: QpetSzQoIErrorBound::Relative {
                    rel: Positive(0.42),
                },
                // data_error_bound: QpetSzDataErrorBound::Absolute {
                //     abs: Positive(f64::MAX),
                // },
                qoi_c: default_qoi_c(),
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
            &QpetSzCompressionMode::SymbolicQuantityOfInterest {
                qoi: String::from("x"),
                qoi_region_size: default_qoi_region_size(),
                qoi_error_bound: QpetSzQoIErrorBound::Absolute {
                    abs: Positive(42.0),
                },
                // data_error_bound: QpetSzDataErrorBound::Absolute {
                //     abs: Positive(f64::MAX),
                // },
                qoi_c: default_qoi_c(),
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
            QpetSzCompressionMode::SymbolicQuantityOfInterest {
                qoi: String::from("x^2"),
                qoi_region_size: default_qoi_region_size(),
                qoi_error_bound: QpetSzQoIErrorBound::Absolute { abs: Positive(0.1) },
                // data_error_bound: QpetSzDataErrorBound::Absolute {
                //     abs: Positive(f64::MAX),
                // },
                qoi_c: default_qoi_c(),
            },
            QpetSzCompressionMode::SymbolicQuantityOfInterest {
                qoi: String::from("x^2"),
                qoi_region_size: default_qoi_region_size(),
                qoi_error_bound: QpetSzQoIErrorBound::Relative {
                    rel: Positive(0.02),
                },
                // data_error_bound: QpetSzDataErrorBound::Absolute {
                //     abs: Positive(f64::MAX),
                // },
                qoi_c: default_qoi_c(),
            },
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
                &QpetSzCompressionMode::SymbolicQuantityOfInterest {
                    qoi: String::from("x"),
                    qoi_region_size: default_qoi_region_size(),
                    qoi_error_bound: QpetSzQoIErrorBound::Absolute {
                        abs: Positive(f64::EPSILON),
                    },
                    // data_error_bound: QpetSzDataErrorBound::Absolute {
                    //     abs: Positive(f64::MAX),
                    // },
                    qoi_c: default_qoi_c(),
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
            &QpetSzCompressionMode::SymbolicQuantityOfInterest {
                qoi: String::from("x^2"),
                qoi_region_size: default_qoi_region_size(),
                qoi_error_bound: QpetSzQoIErrorBound::Absolute { abs: Positive(0.1) },
                // data_error_bound: QpetSzDataErrorBound::Absolute {
                //     abs: Positive(f64::MAX),
                // },
                qoi_c: default_qoi_c(),
            },
        )
        .unwrap();
        let decoded = decompress(&encoded).unwrap();

        assert_eq!(decoded.dtype(), AnyArrayDType::F64);
        assert_eq!(decoded.len(), 64 * 64 * 1);
        assert_eq!(decoded.shape(), &[64, 64, 1]);
    }

    #[test]
    fn log10_decode() {
        let encoded = compress(
            Array::<f32, _>::logspace(2.0, 0.0, 100.0, 721 * 1440)
                .into_shape_clone((721, 1440))
                .unwrap(),
            &QpetSzCompressionMode::SymbolicQuantityOfInterest {
                qoi: String::from("log(x, 10)"),
                qoi_region_size: default_qoi_region_size(),
                qoi_error_bound: QpetSzQoIErrorBound::Absolute {
                    abs: Positive(0.25),
                },
                // data_error_bound: QpetSzDataErrorBound::Absolute {
                //     abs: Positive(f64::MAX),
                // },
                qoi_c: default_qoi_c(),
            },
        )
        .unwrap();
        let decoded = decompress(&encoded).unwrap();

        assert_eq!(decoded.dtype(), AnyArrayDType::F32);
        assert_eq!(decoded.len(), 721 * 1440);
        assert_eq!(decoded.shape(), &[721, 1440]);
    }
}
