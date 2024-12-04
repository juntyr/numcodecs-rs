//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.81.0-blue
//! [repo]: https://github.com/juntyr/numcodecs-rs
//!
//! [Latest Version]: https://img.shields.io/crates/v/numcodecs-sz3
//! [crates.io]: https://crates.io/crates/numcodecs-sz3
//!
//! [Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs-sz3
//! [docs.rs]: https://docs.rs/numcodecs-sz3/
//!
//! [Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/numcodecs-rs/numcodecs_sz3
//!
//! SZ3 codec implementation for the [`numcodecs`] API.

use std::{borrow::Cow, fmt};

use ndarray::{Array, Array1, ArrayBase, Data, Dimension, ShapeError};
use numcodecs::{
    AnyArray, AnyArrayAssignError, AnyArrayDType, AnyArrayView, AnyArrayViewMut, AnyCowArray,
    Codec, StaticCodec, StaticCodecConfig,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;

// Only included to explicitly enable the `no_wasm_shim` feature for
// sz3-sys/Sz3-sys
use ::zstd_sys as _;

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
#[serde(transparent)]
/// Codec providing compression using SZ3
pub struct Sz3Codec {
    /// SZ3 error bound
    pub error: Sz3ErrorBound,
}

/// SZ3 error bound
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "eb_mode")]
#[serde(deny_unknown_fields)]
pub enum Sz3ErrorBound {
    /// Errors are bounded by *both* the absolute and relative error, i.e. by
    /// whichever bound is stricter
    #[serde(rename = "abs-and-rel")]
    AbsoluteAndRelative {
        /// Absolute error bound
        #[serde(rename = "eb_abs")]
        abs: f64,
        /// Relative error bound
        #[serde(rename = "eb_rel")]
        rel: f64,
    },
    /// Errors are bounded by *either* the absolute or relative error, i.e. by
    /// whichever bound is weaker
    #[serde(rename = "abs-or-rel")]
    AbsoluteOrRelative {
        /// Absolute error bound
        #[serde(rename = "eb_abs")]
        abs: f64,
        /// Relative error bound
        #[serde(rename = "eb_rel")]
        rel: f64,
    },
    /// Absolute error bound
    #[serde(rename = "abs")]
    Absolute {
        /// Absolute error bound
        #[serde(rename = "eb_abs")]
        abs: f64,
    },
    /// Relative error bound
    #[serde(rename = "rel")]
    Relative {
        /// Relative error bound
        #[serde(rename = "eb_rel")]
        rel: f64,
    },
    /// Peak signal to noise ratio error bound
    #[serde(rename = "psnr")]
    PS2NR {
        /// Peak signal to noise ratio error bound
        #[serde(rename = "eb_psnr")]
        psnr: f64,
    },
    /// Peak L2 norm error bound
    #[serde(rename = "l2")]
    L2Norm {
        /// Peak L2 norm error bound
        #[serde(rename = "eb_l2")]
        l2: f64,
    },
}

impl Codec for Sz3Codec {
    type Error = Sz3CodecError;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        match data {
            AnyCowArray::I32(data) => Ok(AnyArray::U8(
                Array1::from(compress(data, &self.error)?).into_dyn(),
            )),
            AnyCowArray::I64(data) => Ok(AnyArray::U8(
                Array1::from(compress(data, &self.error)?).into_dyn(),
            )),
            AnyCowArray::F32(data) => Ok(AnyArray::U8(
                Array1::from(compress(data, &self.error)?).into_dyn(),
            )),
            AnyCowArray::F64(data) => Ok(AnyArray::U8(
                Array1::from(compress(data, &self.error)?).into_dyn(),
            )),
            encoded => Err(Sz3CodecError::UnsupportedDtype(encoded.dtype())),
        }
    }

    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
        let AnyCowArray::U8(encoded) = encoded else {
            return Err(Sz3CodecError::EncodedDataNotBytes {
                dtype: encoded.dtype(),
            });
        };

        if !matches!(encoded.shape(), [_]) {
            return Err(Sz3CodecError::EncodedDataNotOneDimensional {
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

impl StaticCodec for Sz3Codec {
    const CODEC_ID: &'static str = "sz3";

    type Config<'de> = Self;

    fn from_config(config: Self::Config<'_>) -> Self {
        config
    }

    fn get_config(&self) -> StaticCodecConfig<Self> {
        StaticCodecConfig::from(self)
    }
}

#[derive(Debug, Error)]
/// Errors that may occur when applying the [`Sz3Codec`].
pub enum Sz3CodecError {
    /// [`Sz3Codec`] does not support the dtype
    #[error("Sz3 does not support the dtype {0}")]
    UnsupportedDtype(AnyArrayDType),
    /// [`Sz3Codec`] failed to encode the header
    #[error("Sz3 failed to encode the header")]
    HeaderEncodeFailed {
        /// Opaque source error
        source: Sz3HeaderError,
    },
    /// [`Sz3Codec`] cannot encode an array of `shape`
    #[error("Sz3 cannot encode an array of shape {shape:?}")]
    InvalidEncodeShape {
        /// Opaque source error
        source: Sz3CodingError,
        /// The invalid shape of the encoded array
        shape: Vec<usize>,
    },
    /// [`Sz3Codec`] failed to encode the data
    #[error("Sz3 failed to encode the data")]
    Sz3EncodeFailed {
        /// Opaque source error
        source: Sz3CodingError,
    },
    /// [`Sz3Codec`] can only decode one-dimensional byte arrays but received
    /// an array of a different dtype
    #[error(
        "Sz3 can only decode one-dimensional byte arrays but received an array of dtype {dtype}"
    )]
    EncodedDataNotBytes {
        /// The unexpected dtype of the encoded array
        dtype: AnyArrayDType,
    },
    /// [`Sz3Codec`] can only decode one-dimensional byte arrays but received
    /// an array of a different shape
    #[error("Sz3 can only decode one-dimensional byte arrays but received a byte array of shape {shape:?}")]
    EncodedDataNotOneDimensional {
        /// The unexpected shape of the encoded array
        shape: Vec<usize>,
    },
    /// [`Sz3Codec`] failed to decode the header
    #[error("Sz3 failed to decode the header")]
    HeaderDecodeFailed {
        /// Opaque source error
        source: Sz3HeaderError,
    },
    /// [`Sz3Codec`] decoded an invalid array shape header which does not fit
    /// the decoded data
    #[error("Sz3 decoded an invalid array shape header which does not fit the decoded data")]
    DecodeInvalidShapeHeader {
        /// Source error
        #[from]
        source: ShapeError,
    },
    /// [`Sz3Codec`] cannot decode into the provided array
    #[error("Sz3 cannot decode into the provided array")]
    MismatchedDecodeIntoArray {
        /// The source of the error
        #[from]
        source: AnyArrayAssignError,
    },
}

#[derive(Debug, Error)]
#[error(transparent)]
/// Opaque error for when encoding or decoding the header fails
pub struct Sz3HeaderError(postcard::Error);

#[derive(Debug, Error)]
#[error(transparent)]
/// Opaque error for when encoding or decoding with SZ3 fails
pub struct Sz3CodingError(sz3::SZ3Error);

#[allow(clippy::needless_pass_by_value)]
/// Compresses the input `data` array using SZ3 with the provided error `bound`
/// configuration.
///
/// # Errors
///
/// Errors with
/// - [`Sz3CodecError::HeaderEncodeFailed`] if encoding the header failed
/// - [`Sz3CodecError::InvalidEncodeShape`] if the array shape is invalid
/// - [`Sz3CodecError::Sz3EncodeFailed`] if encoding failed with an opaque error
pub fn compress<T: Sz3Element, S: Data<Elem = T>, D: Dimension>(
    data: ArrayBase<S, D>,
    bound: &Sz3ErrorBound,
) -> Result<Vec<u8>, Sz3CodecError> {
    let mut encoded = postcard::to_extend(
        &CompressionHeader {
            dtype: <T as Sz3Element>::DTYPE,
            shape: Cow::Borrowed(data.shape()),
        },
        Vec::new(),
    )
    .map_err(|err| Sz3CodecError::HeaderEncodeFailed {
        source: Sz3HeaderError(err),
    })?;

    // sz3::DimensionedDataBuilder cannot handle zero-length dimensions
    if data.is_empty() {
        return Ok(encoded);
    }

    #[allow(clippy::option_if_let_else)]
    let data_cow = if let Some(data) = data.as_slice() {
        Cow::Borrowed(data)
    } else {
        Cow::Owned(data.iter().copied().collect())
    };
    let mut builder = sz3::DimensionedData::build(&data_cow);

    for length in data.shape() {
        // Sz3 ignores dimensions of length 1 and panics on length zero
        // Since they carry no information for Sz3 and we already encode them
        //  in our custom header, we just skip them here
        if *length > 1 {
            builder = builder
                .dim(*length)
                .map_err(|err| Sz3CodecError::InvalidEncodeShape {
                    source: Sz3CodingError(err),
                    shape: data.shape().to_vec(),
                })?;
        }
    }

    if data.len() == 1 {
        // If there is only one element, all dimensions will have been skipped,
        //  so we explicitly encode one dimension of size 1 here
        builder = builder
            .dim(1)
            .map_err(|err| Sz3CodecError::InvalidEncodeShape {
                source: Sz3CodingError(err),
                shape: data.shape().to_vec(),
            })?;
    }

    let data = builder
        .finish()
        .map_err(|err| Sz3CodecError::InvalidEncodeShape {
            source: Sz3CodingError(err),
            shape: data.shape().to_vec(),
        })?;

    let error_bound = match bound {
        Sz3ErrorBound::AbsoluteAndRelative { abs, rel } => sz3::ErrorBound::AbsoluteAndRelative {
            absolute_bound: *abs,
            relative_bound: *rel,
        },
        Sz3ErrorBound::AbsoluteOrRelative { abs, rel } => sz3::ErrorBound::AbsoluteOrRelative {
            absolute_bound: *abs,
            relative_bound: *rel,
        },
        Sz3ErrorBound::Absolute { abs } => sz3::ErrorBound::Absolute(*abs),
        Sz3ErrorBound::Relative { rel } => sz3::ErrorBound::Relative(*rel),
        Sz3ErrorBound::PS2NR { psnr } => sz3::ErrorBound::PSNR(*psnr),
        Sz3ErrorBound::L2Norm { l2 } => sz3::ErrorBound::L2Norm(*l2),
    };

    // FIXME: Sz3 seems to have a UB bug that impacts the last few bytes but is
    //        somehow gone if we use stdio first ... aaaaaaaah
    std::mem::drop(std::io::Read::read(&mut std::io::stdin(), &mut []));

    // TODO: avoid extra allocation here
    let compressed =
        sz3::compress(&data, error_bound).map_err(|err| Sz3CodecError::Sz3EncodeFailed {
            source: Sz3CodingError(err),
        })?;
    encoded.extend_from_slice(&compressed);

    Ok(encoded)
}

/// Decompresses the `encoded` data into an array.
///
/// # Errors
///
/// Errors with
/// - [`Sz3CodecError::HeaderDecodeFailed`] if decoding the header failed
pub fn decompress(encoded: &[u8]) -> Result<AnyArray, Sz3CodecError> {
    let (header, data) =
        postcard::take_from_bytes::<CompressionHeader>(encoded).map_err(|err| {
            Sz3CodecError::HeaderDecodeFailed {
                source: Sz3HeaderError(err),
            }
        })?;

    let decoded = if header.shape.iter().copied().product::<usize>() == 0 {
        match header.dtype {
            Sz3DType::I32 => {
                AnyArray::I32(Array::from_shape_vec(&*header.shape, Vec::new())?.into_dyn())
            }
            Sz3DType::I64 => {
                AnyArray::I64(Array::from_shape_vec(&*header.shape, Vec::new())?.into_dyn())
            }
            Sz3DType::F32 => {
                AnyArray::F32(Array::from_shape_vec(&*header.shape, Vec::new())?.into_dyn())
            }
            Sz3DType::F64 => {
                AnyArray::F64(Array::from_shape_vec(&*header.shape, Vec::new())?.into_dyn())
            }
        }
    } else {
        // TODO: avoid extra allocation here
        match header.dtype {
            Sz3DType::I32 => AnyArray::I32(Array::from_shape_vec(
                &*header.shape,
                Vec::from(sz3::decompress(data).1.data()),
            )?),
            Sz3DType::I64 => AnyArray::I64(Array::from_shape_vec(
                &*header.shape,
                Vec::from(sz3::decompress(data).1.data()),
            )?),
            Sz3DType::F32 => AnyArray::F32(Array::from_shape_vec(
                &*header.shape,
                Vec::from(sz3::decompress(data).1.data()),
            )?),
            Sz3DType::F64 => AnyArray::F64(Array::from_shape_vec(
                &*header.shape,
                Vec::from(sz3::decompress(data).1.data()),
            )?),
        }
    };

    Ok(decoded)
}

/// Array element types which can be compressed with SZ3.
pub trait Sz3Element: Copy + sz3::SZ3Compressible {
    /// The dtype representation of the type
    const DTYPE: Sz3DType;
}

impl Sz3Element for i32 {
    const DTYPE: Sz3DType = Sz3DType::I32;
}

impl Sz3Element for i64 {
    const DTYPE: Sz3DType = Sz3DType::I64;
}

impl Sz3Element for f32 {
    const DTYPE: Sz3DType = Sz3DType::F32;
}

impl Sz3Element for f64 {
    const DTYPE: Sz3DType = Sz3DType::F64;
}

#[derive(Serialize, Deserialize)]
struct CompressionHeader<'a> {
    dtype: Sz3DType,
    #[serde(borrow)]
    shape: Cow<'a, [usize]>,
}

/// Dtypes that SZ3 can compress and decompress
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum Sz3DType {
    #[serde(rename = "i32", alias = "int32")]
    I32,
    #[serde(rename = "i64", alias = "int64")]
    I64,
    #[serde(rename = "f32", alias = "float32")]
    F32,
    #[serde(rename = "f64", alias = "float64")]
    F64,
}

impl fmt::Display for Sz3DType {
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
mod tests {
    use ndarray::ArrayView1;

    use super::*;

    #[test]
    fn zero_length() -> Result<(), Sz3CodecError> {
        let encoded = compress(
            Array::<f32, _>::from_shape_vec([1, 27, 0].as_slice(), vec![])?,
            &Sz3ErrorBound::L2Norm { l2: 27.0 },
        )?;
        let decoded = decompress(&encoded)?;

        assert_eq!(decoded.dtype(), AnyArrayDType::F32);
        assert!(decoded.is_empty());
        assert_eq!(decoded.shape(), &[1, 27, 0]);

        Ok(())
    }

    #[test]
    fn one_dimension() -> Result<(), Sz3CodecError> {
        let data = Array::from_shape_vec([2_usize, 1, 2, 1].as_slice(), vec![1, 2, 3, 4])?;

        let encoded = compress(data.view(), &Sz3ErrorBound::Absolute { abs: 0.0 })?;
        let decoded = decompress(&encoded)?;

        assert_eq!(decoded, AnyArray::I32(data));

        Ok(())
    }

    #[test]
    fn small_state() -> Result<(), Sz3CodecError> {
        for data in [
            &[][..],
            &[0.0],
            &[0.0, 1.0],
            &[0.0, 1.0, 0.0],
            &[0.0, 1.0, 0.0, 1.0],
        ] {
            let encoded = compress(
                ArrayView1::from(data),
                &Sz3ErrorBound::Absolute { abs: 0.0 },
            )?;
            let decoded = decompress(&encoded)?;

            assert_eq!(
                decoded,
                AnyArray::F64(Array1::from_vec(data.to_vec()).into_dyn())
            );
        }

        Ok(())
    }
}
