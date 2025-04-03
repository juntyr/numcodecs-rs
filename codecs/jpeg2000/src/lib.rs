//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.82.0-blue
//! [repo]: https://github.com/juntyr/numcodecs-rs
//!
//! [Latest Version]: https://img.shields.io/crates/v/numcodecs-jpeg2000
//! [crates.io]: https://crates.io/crates/numcodecs-jpeg2000
//!
//! [Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs-jpeg2000
//! [docs.rs]: https://docs.rs/numcodecs-jpeg2000/
//!
//! [Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/numcodecs-rs/numcodecs_jpeg2000
//!
//! JPEG 2000 codec implementation for the [`numcodecs`] API.

#![allow(clippy::multiple_crate_versions)] // embedded-io

#[cfg(test)]
use ::serde_json as _;

use std::borrow::Cow;
use std::fmt;

use ndarray::{Array, Array1, ArrayBase, Axis, Data, Dimension, IxDyn, ShapeError};
use num_traits::identities::Zero;
use numcodecs::{
    AnyArray, AnyArrayAssignError, AnyArrayDType, AnyArrayView, AnyArrayViewMut, AnyCowArray,
    Codec, StaticCodec, StaticCodecConfig,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;

mod ffi;

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
// serde cannot deny unknown fields because of the flatten
#[schemars(deny_unknown_fields)]
/// Codec providing compression using JPEG 2000.
///
/// Arrays that are higher-dimensional than 2D are encoded by compressing each
/// 2D slice with JPEG 2000 independently. Specifically, the array's shape is
/// interpreted as `[.., height, width]`. If you want to compress 2D slices
/// along two different axes, you can swizzle the array axes beforehand.
pub struct Jpeg2000Codec {
    /// JPEG 2000 compression mode
    #[serde(flatten)]
    pub mode: Jpeg2000CompressionMode,
}

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
/// JPEG 2000 compression mode
#[serde(tag = "mode")]
pub enum Jpeg2000CompressionMode {
    /// Peak signal-to-noise ratio
    #[serde(rename = "psnr")]
    PSNR {
        /// Peak signal-to-noise ratio
        psnr: f32,
    },
    /// Compression rate
    #[serde(rename = "rate")]
    Rate {
        /// Compression rate, e.g. `10.0` for x10 compression
        rate: f32,
    },
    /// Lossless compression
    #[serde(rename = "lossless")]
    Lossless,
}

impl Codec for Jpeg2000Codec {
    type Error = Jpeg2000CodecError;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        match data {
            AnyCowArray::I8(data) => Ok(AnyArray::U8(
                Array1::from(compress(data, &self.mode)?).into_dyn(),
            )),
            AnyCowArray::U8(data) => Ok(AnyArray::U8(
                Array1::from(compress(data, &self.mode)?).into_dyn(),
            )),
            AnyCowArray::I16(data) => Ok(AnyArray::U8(
                Array1::from(compress(data, &self.mode)?).into_dyn(),
            )),
            AnyCowArray::U16(data) => Ok(AnyArray::U8(
                Array1::from(compress(data, &self.mode)?).into_dyn(),
            )),
            AnyCowArray::I32(data) => Ok(AnyArray::U8(
                Array1::from(compress(data, &self.mode)?).into_dyn(),
            )),
            AnyCowArray::U32(data) => Ok(AnyArray::U8(
                Array1::from(compress(data, &self.mode)?).into_dyn(),
            )),
            AnyCowArray::I64(data) => Ok(AnyArray::U8(
                Array1::from(compress(data, &self.mode)?).into_dyn(),
            )),
            AnyCowArray::U64(data) => Ok(AnyArray::U8(
                Array1::from(compress(data, &self.mode)?).into_dyn(),
            )),
            encoded => Err(Jpeg2000CodecError::UnsupportedDtype(encoded.dtype())),
        }
    }

    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
        let AnyCowArray::U8(encoded) = encoded else {
            return Err(Jpeg2000CodecError::EncodedDataNotBytes {
                dtype: encoded.dtype(),
            });
        };

        if !matches!(encoded.shape(), [_]) {
            return Err(Jpeg2000CodecError::EncodedDataNotOneDimensional {
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

impl StaticCodec for Jpeg2000Codec {
    const CODEC_ID: &'static str = "jpeg2000.rs";

    type Config<'de> = Self;

    fn from_config(config: Self::Config<'_>) -> Self {
        config
    }

    fn get_config(&self) -> StaticCodecConfig<Self> {
        StaticCodecConfig::from(self)
    }
}

#[derive(Debug, Error)]
/// Errors that may occur when applying the [`Jpeg2000Codec`].
pub enum Jpeg2000CodecError {
    /// [`Jpeg2000Codec`] does not support the dtype
    #[error("Jpeg2000 does not support the dtype {0}")]
    UnsupportedDtype(AnyArrayDType),
    /// [`Jpeg2000Codec`] failed to encode the header
    #[error("Jpeg2000 failed to encode the header")]
    HeaderEncodeFailed {
        /// Opaque source error
        source: Jpeg2000HeaderError,
    },
    /// [`Jpeg2000Codec`] failed to encode the data
    #[error("Jpeg2000 failed to encode the data")]
    Jpeg2000EncodeFailed {
        /// Opaque source error
        source: Jpeg2000CodingError,
    },
    /// [`Jpeg2000Codec`] failed to encode a slice
    #[error("Jpeg2000 failed to encode a slice")]
    SliceEncodeFailed {
        /// Opaque source error
        source: Jpeg2000SliceError,
    },
    /// [`Jpeg2000Codec`] can only decode one-dimensional byte arrays but received
    /// an array of a different dtype
    #[error(
        "Jpeg2000 can only decode one-dimensional byte arrays but received an array of dtype {dtype}"
    )]
    EncodedDataNotBytes {
        /// The unexpected dtype of the encoded array
        dtype: AnyArrayDType,
    },
    /// [`Jpeg2000Codec`] can only decode one-dimensional byte arrays but received
    /// an array of a different shape
    #[error("Jpeg2000 can only decode one-dimensional byte arrays but received a byte array of shape {shape:?}")]
    EncodedDataNotOneDimensional {
        /// The unexpected shape of the encoded array
        shape: Vec<usize>,
    },
    /// [`Jpeg2000Codec`] failed to decode the header
    #[error("Jpeg2000 failed to decode the header")]
    HeaderDecodeFailed {
        /// Opaque source error
        source: Jpeg2000HeaderError,
    },
    /// [`Jpeg2000Codec`] failed to decode a slice
    #[error("Jpeg2000 failed to decode a slice")]
    SliceDecodeFailed {
        /// Opaque source error
        source: Jpeg2000SliceError,
    },
    /// [`Jpeg2000Codec`] failed to decode from an excessive number of slices
    #[error("Jpeg2000 failed to decode from an excessive number of slices")]
    DecodeTooManySlices,
    /// [`Jpeg2000Codec`] failed to decode the data
    #[error("Jpeg2000 failed to decode the data")]
    Jpeg2000DecodeFailed {
        /// Opaque source error
        source: Jpeg2000CodingError,
    },
    /// [`Jpeg2000Codec`] decoded into an invalid shape not matching the data size
    #[error("Jpeg2000 decoded into an invalid shape not matching the data size")]
    DecodeInvalidShape {
        /// The source of the error
        source: ShapeError,
    },
    /// [`Jpeg2000Codec`] cannot decode into the provided array
    #[error("Jpeg2000Codec cannot decode into the provided array")]
    MismatchedDecodeIntoArray {
        /// The source of the error
        #[from]
        source: AnyArrayAssignError,
    },
}

#[derive(Debug, Error)]
#[error(transparent)]
/// Opaque error for when encoding or decoding the header fails
pub struct Jpeg2000HeaderError(postcard::Error);

#[derive(Debug, Error)]
#[error(transparent)]
/// Opaque error for when encoding or decoding a slice fails
pub struct Jpeg2000SliceError(postcard::Error);

#[derive(Debug, Error)]
#[error(transparent)]
/// Opaque error for when encoding or decoding with JPEG 2000 fails
pub struct Jpeg2000CodingError(ffi::Jpeg2000Error);

/// Compress the `data` array using JPEG 2000 with the provided `mode`.
///
/// # Errors
///
/// Errors with
/// - [`Jpeg2000CodecError::HeaderEncodeFailed`] if encoding the header failed
/// - [`Jpeg2000CodecError::Jpeg2000EncodeFailed`] if encoding with JPEG 2000
///   failed
/// - [`Jpeg2000CodecError::SliceEncodeFailed`] if encoding a slice failed
pub fn compress<T: Jpeg2000Element, S: Data<Elem = T>, D: Dimension>(
    data: ArrayBase<S, D>,
    mode: &Jpeg2000CompressionMode,
) -> Result<Vec<u8>, Jpeg2000CodecError> {
    let mut encoded = postcard::to_extend(
        &CompressionHeader {
            dtype: T::DTYPE,
            shape: Cow::Borrowed(data.shape()),
        },
        Vec::new(),
    )
    .map_err(|err| Jpeg2000CodecError::HeaderEncodeFailed {
        source: Jpeg2000HeaderError(err),
    })?;

    // JPEG 2000 cannot handle zero-length dimensions
    if data.is_empty() {
        return Ok(encoded);
    }

    let mut encoded_slice = Vec::new();

    let mut chunk_size = Vec::from(data.shape());
    let (width, height) = match *chunk_size.as_mut_slice() {
        [ref mut rest @ .., height, width] => {
            for r in rest {
                *r = 1;
            }
            (width, height)
        }
        [width] => (width, 1),
        [] => (1, 1),
    };

    for slice in data.into_dyn().exact_chunks(chunk_size.as_slice()) {
        encoded_slice.clear();

        ffi::encode_into(
            slice.iter().copied(),
            width,
            height,
            match mode {
                Jpeg2000CompressionMode::PSNR { psnr } => ffi::Jpeg2000CompressionMode::PSNR(*psnr),
                Jpeg2000CompressionMode::Rate { rate } => ffi::Jpeg2000CompressionMode::Rate(*rate),
                Jpeg2000CompressionMode::Lossless => ffi::Jpeg2000CompressionMode::Lossless,
            },
            &mut encoded_slice,
        )
        .map_err(|err| Jpeg2000CodecError::Jpeg2000EncodeFailed {
            source: Jpeg2000CodingError(err),
        })?;

        encoded = postcard::to_extend(encoded_slice.as_slice(), encoded).map_err(|err| {
            Jpeg2000CodecError::SliceEncodeFailed {
                source: Jpeg2000SliceError(err),
            }
        })?;
    }

    Ok(encoded)
}

/// Decompress the `encoded` data into an array using JPEG 2000.
///
/// # Errors
///
/// Errors with
/// - [`Jpeg2000CodecError::HeaderDecodeFailed`] if decoding the header failed
/// - [`Jpeg2000CodecError::SliceDecodeFailed`] if decoding a slice failed
/// - [`Jpeg2000CodecError::Jpeg2000DecodeFailed`] if decoding with JPEG 2000
///   failed
/// - [`Jpeg2000CodecError::DecodeInvalidShape`] if the encoded data decodes to
///   an unexpected shape
/// - [`Jpeg2000CodecError::DecodeTooManySlices`] if the encoded data contains
///   too many slices
pub fn decompress(encoded: &[u8]) -> Result<AnyArray, Jpeg2000CodecError> {
    fn decompress_typed<T: Jpeg2000Element>(
        mut encoded: &[u8],
        shape: &[usize],
    ) -> Result<Array<T, IxDyn>, Jpeg2000CodecError> {
        let mut decoded = Array::<T, _>::zeros(shape);

        let mut chunk_size = Vec::from(shape);
        let (width, height) = match *chunk_size.as_mut_slice() {
            [ref mut rest @ .., height, width] => {
                for r in rest {
                    *r = 1;
                }
                (width, height)
            }
            [width] => (width, 1),
            [] => (1, 1),
        };

        for mut slice in decoded.exact_chunks_mut(chunk_size.as_slice()) {
            let (encoded_slice, rest) =
                postcard::take_from_bytes::<Cow<[u8]>>(encoded).map_err(|err| {
                    Jpeg2000CodecError::SliceDecodeFailed {
                        source: Jpeg2000SliceError(err),
                    }
                })?;
            encoded = rest;

            let (decoded_slice, (_width, _height)) =
                ffi::decode::<T>(&encoded_slice).map_err(|err| {
                    Jpeg2000CodecError::Jpeg2000DecodeFailed {
                        source: Jpeg2000CodingError(err),
                    }
                })?;
            let mut decoded_slice = Array::from_shape_vec((height, width), decoded_slice)
                .map_err(|source| Jpeg2000CodecError::DecodeInvalidShape { source })?
                .into_dyn();

            while decoded_slice.ndim() > shape.len() {
                decoded_slice = decoded_slice.remove_axis(Axis(0));
            }

            slice.assign(&decoded_slice);
        }

        if !encoded.is_empty() {
            return Err(Jpeg2000CodecError::DecodeTooManySlices);
        }

        Ok(decoded)
    }

    let (header, encoded) =
        postcard::take_from_bytes::<CompressionHeader>(encoded).map_err(|err| {
            Jpeg2000CodecError::HeaderDecodeFailed {
                source: Jpeg2000HeaderError(err),
            }
        })?;

    // Return empty data for zero-size arrays
    if header.shape.iter().copied().product::<usize>() == 0 {
        return match header.dtype {
            Jpeg2000DType::I8 => Ok(AnyArray::I8(Array::zeros(&*header.shape))),
            Jpeg2000DType::U8 => Ok(AnyArray::U8(Array::zeros(&*header.shape))),
            Jpeg2000DType::I16 => Ok(AnyArray::I16(Array::zeros(&*header.shape))),
            Jpeg2000DType::U16 => Ok(AnyArray::U16(Array::zeros(&*header.shape))),
            Jpeg2000DType::I32 => Ok(AnyArray::I32(Array::zeros(&*header.shape))),
            Jpeg2000DType::U32 => Ok(AnyArray::U32(Array::zeros(&*header.shape))),
            Jpeg2000DType::I64 => Ok(AnyArray::I64(Array::zeros(&*header.shape))),
            Jpeg2000DType::U64 => Ok(AnyArray::U64(Array::zeros(&*header.shape))),
        };
    }

    match header.dtype {
        Jpeg2000DType::I8 => Ok(AnyArray::I8(decompress_typed(encoded, &header.shape)?)),
        Jpeg2000DType::U8 => Ok(AnyArray::U8(decompress_typed(encoded, &header.shape)?)),
        Jpeg2000DType::I16 => Ok(AnyArray::I16(decompress_typed(encoded, &header.shape)?)),
        Jpeg2000DType::U16 => Ok(AnyArray::U16(decompress_typed(encoded, &header.shape)?)),
        Jpeg2000DType::I32 => Ok(AnyArray::I32(decompress_typed(encoded, &header.shape)?)),
        Jpeg2000DType::U32 => Ok(AnyArray::U32(decompress_typed(encoded, &header.shape)?)),
        Jpeg2000DType::I64 => Ok(AnyArray::I64(decompress_typed(encoded, &header.shape)?)),
        Jpeg2000DType::U64 => Ok(AnyArray::U64(decompress_typed(encoded, &header.shape)?)),
    }
}

/// Array element types which can be compressed with JPEG 2000.
pub trait Jpeg2000Element: ffi::Jpeg2000Element + Zero {
    /// The dtype representation of the type
    const DTYPE: Jpeg2000DType;
}

impl Jpeg2000Element for i8 {
    const DTYPE: Jpeg2000DType = Jpeg2000DType::I8;
}
impl Jpeg2000Element for u8 {
    const DTYPE: Jpeg2000DType = Jpeg2000DType::U8;
}
impl Jpeg2000Element for i16 {
    const DTYPE: Jpeg2000DType = Jpeg2000DType::I16;
}
impl Jpeg2000Element for u16 {
    const DTYPE: Jpeg2000DType = Jpeg2000DType::U16;
}
impl Jpeg2000Element for i32 {
    const DTYPE: Jpeg2000DType = Jpeg2000DType::I32;
}
impl Jpeg2000Element for u32 {
    const DTYPE: Jpeg2000DType = Jpeg2000DType::U32;
}
impl Jpeg2000Element for i64 {
    const DTYPE: Jpeg2000DType = Jpeg2000DType::I64;
}
impl Jpeg2000Element for u64 {
    const DTYPE: Jpeg2000DType = Jpeg2000DType::U64;
}

#[derive(Serialize, Deserialize)]
struct CompressionHeader<'a> {
    dtype: Jpeg2000DType,
    #[serde(borrow)]
    shape: Cow<'a, [usize]>,
}

/// Dtypes that JPEG 2000 can compress and decompress
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
#[expect(missing_docs)]
pub enum Jpeg2000DType {
    #[serde(rename = "i8", alias = "int8")]
    I8,
    #[serde(rename = "u8", alias = "uint8")]
    U8,
    #[serde(rename = "i16", alias = "int16")]
    I16,
    #[serde(rename = "u16", alias = "uint16")]
    U16,
    #[serde(rename = "i32", alias = "int32")]
    I32,
    #[serde(rename = "u32", alias = "uint32")]
    U32,
    #[serde(rename = "i64", alias = "int64")]
    I64,
    #[serde(rename = "u64", alias = "uint64")]
    U64,
}

impl fmt::Display for Jpeg2000DType {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.write_str(match self {
            Self::I8 => "i8",
            Self::U8 => "u8",
            Self::I16 => "i16",
            Self::U16 => "u16",
            Self::I32 => "i32",
            Self::U32 => "u32",
            Self::I64 => "i64",
            Self::U64 => "u64",
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
        std::mem::drop(simple_logger::init());

        let encoded = compress(
            Array::<i16, _>::from_shape_vec([3, 0], vec![]).unwrap(),
            &Jpeg2000CompressionMode::PSNR { psnr: 42.0 },
        )
        .unwrap();
        let decoded = decompress(&encoded).unwrap();

        assert_eq!(decoded.dtype(), AnyArrayDType::I16);
        assert!(decoded.is_empty());
        assert_eq!(decoded.shape(), &[3, 0]);
    }

    #[test]
    fn small_2d() {
        std::mem::drop(simple_logger::init());

        let encoded = compress(
            Array::<i16, _>::from_shape_vec([1, 1], vec![42]).unwrap(),
            &Jpeg2000CompressionMode::PSNR { psnr: 42.0 },
        )
        .unwrap();
        let decoded = decompress(&encoded).unwrap();

        assert_eq!(decoded.dtype(), AnyArrayDType::I16);
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded.shape(), &[1, 1]);
    }

    #[test]
    fn small_lossless_types() {
        macro_rules! check {
            ($T:ident($t:ident)) => {
                check! { $T($t,$t::MIN,$t::MAX) }
            };
            ($T:ident($t:ident,$min:expr,$max:expr)) => {
                let data = Array::<$t, _>::from_shape_vec([4, 1], vec![$min, 0, 42, $max]).unwrap();

                let encoded = compress(
                    data.view(),
                    &Jpeg2000CompressionMode::Lossless,
                )
                .unwrap();
                let decoded = decompress(&encoded).unwrap();

                assert_eq!(decoded.len(), 4);
                assert_eq!(decoded.shape(), &[4, 1]);
                assert_eq!(decoded, AnyArray::$T(data.into_dyn()));
            };
            ($($T:ident($($tt:tt),*)),*) => {
                $(check! { $T($($tt),*) })*
            };
        }

        check! {
            I8(i8), U8(u8), I16(i16), U16(u16),
            I32(i32,(i32::MIN/(1<<7)),(i32::MAX/(1<<7))),
            U32(u32,(u32::MIN),(u32::MAX/(1<<7))),
            I64(i64,(i64::MIN/(1<<(32+7))),(i64::MAX/(1<<(32+7)))),
            U64(u64,(u64::MIN),(u64::MAX/(1<<(32+7))))
        }
    }

    #[test]
    fn out_of_range() {
        macro_rules! check {
            ($T:ident($t:ident,$($v:expr),*)) => {
                $(
                    let data = Array::<$t, _>::from_shape_vec([1, 1], vec![$v]).unwrap();
                    compress(
                        data.view(),
                        &Jpeg2000CompressionMode::Lossless,
                    )
                    .unwrap_err();
                )*
            };
            ($($T:ident($($tt:tt),*)),*) => {
                $(check! { $T($($tt),*) })*
            };
        }

        check! {
            I32(i32,(i32::MIN),(i32::MAX)), U32(u32,(u32::MAX)),
            I64(i64,(i64::MIN),(i64::MAX)), U64(u64,(u64::MAX))
        }
    }

    #[test]
    fn large_2d() {
        std::mem::drop(simple_logger::init());

        let encoded = compress(
            Array::<i16, _>::zeros((64, 64)),
            &Jpeg2000CompressionMode::PSNR { psnr: 42.0 },
        )
        .unwrap();
        let decoded = decompress(&encoded).unwrap();

        assert_eq!(decoded.dtype(), AnyArrayDType::I16);
        assert_eq!(decoded.len(), 64 * 64);
        assert_eq!(decoded.shape(), &[64, 64]);
    }

    #[test]
    fn all_modes() {
        std::mem::drop(simple_logger::init());

        for mode in [
            Jpeg2000CompressionMode::PSNR { psnr: 42.0 },
            Jpeg2000CompressionMode::Rate { rate: 5.0 },
            Jpeg2000CompressionMode::Lossless,
        ] {
            let encoded = compress(Array::<i16, _>::zeros((64, 64)), &mode).unwrap();
            let decoded = decompress(&encoded).unwrap();

            assert_eq!(decoded.dtype(), AnyArrayDType::I16);
            assert_eq!(decoded.len(), 64 * 64);
            assert_eq!(decoded.shape(), &[64, 64]);
        }
    }

    #[test]
    fn many_dimensions() {
        std::mem::drop(simple_logger::init());

        for data in [
            Array::<i16, Ix0>::from_shape_vec([], vec![42])
                .unwrap()
                .into_dyn(),
            Array::<i16, Ix1>::from_shape_vec([2], vec![1, 2])
                .unwrap()
                .into_dyn(),
            Array::<i16, Ix2>::from_shape_vec([2, 2], vec![1, 2, 3, 4])
                .unwrap()
                .into_dyn(),
            Array::<i16, Ix3>::from_shape_vec([2, 2, 2], vec![1, 2, 3, 4, 5, 6, 7, 8])
                .unwrap()
                .into_dyn(),
            Array::<i16, Ix4>::from_shape_vec(
                [2, 2, 2, 2],
                vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            )
            .unwrap()
            .into_dyn(),
        ] {
            let encoded = compress(data.view(), &Jpeg2000CompressionMode::Lossless).unwrap();
            let decoded = decompress(&encoded).unwrap();

            assert_eq!(decoded, AnyArray::I16(data));
        }
    }
}
