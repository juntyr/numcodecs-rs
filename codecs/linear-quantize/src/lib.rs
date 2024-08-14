#![allow(clippy::multiple_crate_versions)] // FIXME: twofloat -> hexf -> syn 1.0

//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.76.0-blue
//! [repo]: https://github.com/juntyr/numcodecs-rs
//!
//! [Latest Version]: https://img.shields.io/crates/v/numcodecs-linear-quantize
//! [crates.io]: https://crates.io/crates/numcodecs-linear-quantize
//!
//! [Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs-linear-quantize
//! [docs.rs]: https://docs.rs/numcodecs-linear-quantize/
//!
//! [Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/numcodecs-rs/numcodecs_linear_quantize
//!
//! Linear Quantization codec implementation for the [`numcodecs`] API.

use std::{borrow::Cow, fmt};

use ndarray::{Array, Array1, ArrayBase, ArrayD, ArrayViewMutD, Data, Dimension, ShapeError};
use numcodecs::{
    AnyArray, AnyArrayDType, AnyArrayView, AnyArrayViewMut, AnyCowArray, Codec, StaticCodec,
};
use serde::{de::DeserializeOwned, Deserialize, Deserializer, Serialize, Serializer};
use serde_repr::{Deserialize_repr, Serialize_repr};
use thiserror::Error;
use twofloat::TwoFloat;

#[derive(Clone, Serialize, Deserialize)]
/// Lossy codec to reduce the precision of floating point data.
///
/// The data is quantized to unsigned integers of the best-fitting type.
/// The range and shape of the input data is stored in-band.
pub struct LinearQuantizeCodec {
    /// Dtype of the decoded data
    pub dtype: LinearQuantizeDType,
    /// Binary precision of the encoded data where `bits = log2(bins)`
    pub bits: LinearQuantizeBins,
}

/// Data types which the [`LinearQuantizeCodec`] can quantize
#[derive(Copy, Clone, Debug, serde::Serialize, serde::Deserialize)]
#[allow(missing_docs)]
pub enum LinearQuantizeDType {
    F32,
    F64,
}

impl fmt::Display for LinearQuantizeDType {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.write_str(match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
        })
    }
}

/// Number of bins for quantization, written in base-2 scientific notation.
/// 
/// The binary `#[repr(u8)]` value of each variant is equivalent to the binary
/// logarithm of the number of bins, i.e. the binary precision or the number of
/// bits used.
#[derive(Copy, Clone, Serialize_repr, Deserialize_repr)]
#[repr(u8)]
#[rustfmt::skip]
#[allow(missing_docs)]
pub enum LinearQuantizeBins {
    _1B1 = 1, _1B2, _1B3, _1B4, _1B5, _1B6, _1B7, _1B8,
    _1B9, _1B10, _1B11, _1B12, _1B13, _1B14, _1B15, _1B16,
    _1B17, _1B18, _1B19, _1B20, _1B21, _1B22, _1B23, _1B24,
    _1B25, _1B26, _1B27, _1B28, _1B29, _1B30, _1B31, _1B32,
    _1B33, _1B34, _1B35, _1B36, _1B37, _1B38, _1B39, _1B40,
    _1B41, _1B42, _1B43, _1B44, _1B45, _1B46, _1B47, _1B48,
    _1B49, _1B50, _1B51, _1B52, _1B53, _1B54, _1B55, _1B56,
    _1B57, _1B58, _1B59, _1B60, _1B61, _1B62, _1B63, _1B64,
}

impl Codec for LinearQuantizeCodec {
    type Error = LinearQuantizeCodecError;

    #[allow(clippy::too_many_lines)]
    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        let encoded = match (&data, self.dtype) {
            (AnyCowArray::F32(data), LinearQuantizeDType::F32) => match self.bits as u8 {
                bits @ ..=8 => AnyArray::U8(
                    Array1::from_vec(quantize(data, |x| {
                        let max = f32::from(u8::MAX >> (8 - bits));
                        let x = (x * f32::scale_for_bits(bits)).clamp(0.0, max);
                        #[allow(unsafe_code)]
                        // Safety: x is clamped beforehand
                        unsafe {
                            x.to_int_unchecked::<u8>()
                        }
                    })?)
                    .into_dyn(),
                ),
                bits @ 9..=16 => AnyArray::U16(
                    Array1::from_vec(quantize(data, |x| {
                        let max = f32::from(u16::MAX >> (16 - bits));
                        let x = (x * f32::scale_for_bits(bits)).clamp(0.0, max);
                        #[allow(unsafe_code)]
                        // Safety: x is clamped beforehand
                        unsafe {
                            x.to_int_unchecked::<u16>()
                        }
                    })?)
                    .into_dyn(),
                ),
                bits @ 17..=32 => AnyArray::U32(
                    Array1::from_vec(quantize(data, |x| {
                        // we need to use f64 here to have sufficient precision
                        let max = f64::from(u32::MAX >> (32 - bits));
                        let x = (f64::from(x) * f64::scale_for_bits(bits)).clamp(0.0, max);
                        #[allow(unsafe_code)]
                        // Safety: x is clamped beforehand
                        unsafe {
                            x.to_int_unchecked::<u32>()
                        }
                    })?)
                    .into_dyn(),
                ),
                bits @ 33.. => AnyArray::U64(
                    Array1::from_vec(quantize(data, |x| {
                        // we need to use TwoFloat here to have sufficient precision
                        let max = TwoFloat::from(u64::MAX >> (64 - bits));
                        let x = (TwoFloat::from(x) * f64::scale_for_bits(bits))
                            .max(TwoFloat::from(0.0))
                            .min(max);
                        #[allow(unsafe_code)]
                        // Safety: x is clamped beforehand
                        unsafe {
                            u64::try_from(x).unwrap_unchecked()
                        }
                    })?)
                    .into_dyn(),
                ),
            },
            (AnyCowArray::F64(data), LinearQuantizeDType::F64) => match self.bits as u8 {
                bits @ ..=8 => AnyArray::U8(
                    Array1::from_vec(quantize(data, |x| {
                        let max = f64::from(u8::MAX >> (8 - bits));
                        let x = (x * f64::scale_for_bits(bits)).clamp(0.0, max);
                        #[allow(unsafe_code)]
                        // Safety: x is clamped beforehand
                        unsafe {
                            x.to_int_unchecked::<u8>()
                        }
                    })?)
                    .into_dyn(),
                ),
                bits @ 9..=16 => AnyArray::U16(
                    Array1::from_vec(quantize(data, |x| {
                        let max = f64::from(u16::MAX >> (16 - bits));
                        let x = (x * f64::scale_for_bits(bits)).clamp(0.0, max);
                        #[allow(unsafe_code)]
                        // Safety: x is clamped beforehand
                        unsafe {
                            x.to_int_unchecked::<u16>()
                        }
                    })?)
                    .into_dyn(),
                ),
                bits @ 17..=32 => AnyArray::U32(
                    Array1::from_vec(quantize(data, |x| {
                        let max = f64::from(u32::MAX >> (32 - bits));
                        let x = (x * f64::scale_for_bits(bits)).clamp(0.0, max);
                        #[allow(unsafe_code)]
                        // Safety: x is clamped beforehand
                        unsafe {
                            x.to_int_unchecked::<u32>()
                        }
                    })?)
                    .into_dyn(),
                ),
                bits @ 33.. => AnyArray::U64(
                    Array1::from_vec(quantize(data, |x| {
                        // we need to use TwoFloat here to have sufficient precision
                        let max = TwoFloat::from(u64::MAX >> (64 - bits));
                        let x = (TwoFloat::from(x) * f64::scale_for_bits(bits))
                            .max(TwoFloat::from(0.0))
                            .min(max);
                        #[allow(unsafe_code)]
                        // Safety: x is clamped beforehand
                        unsafe {
                            u64::try_from(x).unwrap_unchecked()
                        }
                    })?)
                    .into_dyn(),
                ),
            },
            (data, dtype) => {
                return Err(LinearQuantizeCodecError::MismatchedEncodeDType {
                    configured: dtype,
                    provided: data.dtype(),
                });
            }
        };

        Ok(encoded)
    }

    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
        #[allow(clippy::option_if_let_else)]
        fn as_standard_order<T: Copy, S: Data<Elem = T>, D: Dimension>(
            array: &ArrayBase<S, D>,
        ) -> Cow<[T]> {
            if let Some(data) = array.as_slice() {
                Cow::Borrowed(data)
            } else {
                Cow::Owned(array.iter().copied().collect())
            }
        }

        if !matches!(encoded.shape(), [_]) {
            return Err(LinearQuantizeCodecError::EncodedDataNotOneDimensional {
                shape: encoded.shape().to_vec(),
            });
        }

        let decoded = match (&encoded, self.dtype) {
            (AnyCowArray::U8(encoded), LinearQuantizeDType::F32) => {
                AnyArray::F32(reconstruct(&as_standard_order(encoded), |x| {
                    f32::from(x) / f32::scale_for_bits(self.bits as u8)
                })?)
            }
            (AnyCowArray::U16(encoded), LinearQuantizeDType::F32) => {
                AnyArray::F32(reconstruct(&as_standard_order(encoded), |x| {
                    f32::from(x) / f32::scale_for_bits(self.bits as u8)
                })?)
            }
            (AnyCowArray::U32(encoded), LinearQuantizeDType::F32) => {
                AnyArray::F32(reconstruct(&as_standard_order(encoded), |x| {
                    // we need to use f64 here to have sufficient precision
                    let x = f64::from(x) / f64::scale_for_bits(self.bits as u8);
                    #[allow(clippy::cast_possible_truncation)]
                    let x = x as f32;
                    x
                })?)
            }
            (AnyCowArray::U64(encoded), LinearQuantizeDType::F32) => {
                AnyArray::F32(reconstruct(&as_standard_order(encoded), |x| {
                    // we need to use TwoFloat here to have sufficient precision
                    let x = TwoFloat::from(x) / f64::scale_for_bits(self.bits as u8);
                    f32::from(x)
                })?)
            }
            (AnyCowArray::U8(encoded), LinearQuantizeDType::F64) => {
                AnyArray::F64(reconstruct(&as_standard_order(encoded), |x| {
                    f64::from(x) / f64::scale_for_bits(self.bits as u8)
                })?)
            }
            (AnyCowArray::U16(encoded), LinearQuantizeDType::F64) => {
                AnyArray::F64(reconstruct(&as_standard_order(encoded), |x| {
                    f64::from(x) / f64::scale_for_bits(self.bits as u8)
                })?)
            }
            (AnyCowArray::U32(encoded), LinearQuantizeDType::F64) => {
                AnyArray::F64(reconstruct(&as_standard_order(encoded), |x| {
                    f64::from(x) / f64::scale_for_bits(self.bits as u8)
                })?)
            }
            (AnyCowArray::U64(encoded), LinearQuantizeDType::F64) => {
                AnyArray::F64(reconstruct(&as_standard_order(encoded), |x| {
                    // we need to use TwoFloat here to have sufficient precision
                    let x = TwoFloat::from(x) / f64::scale_for_bits(self.bits as u8);
                    f64::from(x)
                })?)
            }
            (encoded, _dtype) => {
                return Err(LinearQuantizeCodecError::InvalidEncodedDType {
                    dtype: encoded.dtype(),
                })
            }
        };

        Ok(decoded)
    }

    fn decode_into(
        &self,
        encoded: AnyArrayView,
        decoded: AnyArrayViewMut,
    ) -> Result<(), Self::Error> {
        fn as_standard_order<T: Copy, S: Data<Elem = T>, D: Dimension>(
            array: &ArrayBase<S, D>,
        ) -> Cow<[T]> {
            #[allow(clippy::option_if_let_else)]
            if let Some(data) = array.as_slice() {
                Cow::Borrowed(data)
            } else {
                Cow::Owned(array.iter().copied().collect())
            }
        }

        if !matches!(encoded.shape(), [_]) {
            return Err(LinearQuantizeCodecError::EncodedDataNotOneDimensional {
                shape: encoded.shape().to_vec(),
            });
        }

        match (decoded, self.dtype) {
            (AnyArrayViewMut::F32(decoded), LinearQuantizeDType::F32) => {
                match &encoded {
                    AnyArrayView::U8(encoded) => {
                        reconstruct_into(&as_standard_order(encoded), decoded, |x| {
                            f32::from(x) / f32::scale_for_bits(self.bits as u8)
                        })
                    }
                    AnyArrayView::U16(encoded) => {
                        reconstruct_into(&as_standard_order(encoded), decoded, |x| {
                            f32::from(x) / f32::scale_for_bits(self.bits as u8)
                        })
                    }
                    AnyArrayView::U32(encoded) => {
                        reconstruct_into(&as_standard_order(encoded), decoded, |x| {
                            // we need to use f64 here to have sufficient precision
                            let x = f64::from(x) / f64::scale_for_bits(self.bits as u8);
                            #[allow(clippy::cast_possible_truncation)]
                            let x = x as f32;
                            x
                        })
                    }
                    AnyArrayView::U64(encoded) => {
                        reconstruct_into(&as_standard_order(encoded), decoded, |x| {
                            // we need to use TwoFloat here to have sufficient precision
                            let x = TwoFloat::from(x) / f64::scale_for_bits(self.bits as u8);
                            f32::from(x)
                        })
                    }
                    encoded => {
                        return Err(LinearQuantizeCodecError::InvalidEncodedDType {
                            dtype: encoded.dtype(),
                        })
                    }
                }
            }
            (AnyArrayViewMut::F64(decoded), LinearQuantizeDType::F64) => {
                match &encoded {
                    AnyArrayView::U8(encoded) => {
                        reconstruct_into(&as_standard_order(encoded), decoded, |x| {
                            f64::from(x) / f64::scale_for_bits(self.bits as u8)
                        })
                    }
                    AnyArrayView::U16(encoded) => {
                        reconstruct_into(&as_standard_order(encoded), decoded, |x| {
                            f64::from(x) / f64::scale_for_bits(self.bits as u8)
                        })
                    }
                    AnyArrayView::U32(encoded) => {
                        reconstruct_into(&as_standard_order(encoded), decoded, |x| {
                            f64::from(x) / f64::scale_for_bits(self.bits as u8)
                        })
                    }
                    AnyArrayView::U64(encoded) => {
                        reconstruct_into(&as_standard_order(encoded), decoded, |x| {
                            // we need to use TwoFloat here to have sufficient precision
                            let x = TwoFloat::from(x) / f64::scale_for_bits(self.bits as u8);
                            f64::from(x)
                        })
                    }
                    encoded => {
                        return Err(LinearQuantizeCodecError::InvalidEncodedDType {
                            dtype: encoded.dtype(),
                        })
                    }
                }
            }
            (decoded, dtype) => {
                return Err(LinearQuantizeCodecError::MismatchedDecodeIntoDtype {
                    configured: dtype,
                    provided: decoded.dtype(),
                })
            }
        }?;

        Ok(())
    }

    fn get_config<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.serialize(serializer)
    }
}

impl StaticCodec for LinearQuantizeCodec {
    const CODEC_ID: &'static str = "linear-quantize";

    fn from_config<'de, D: Deserializer<'de>>(config: D) -> Result<Self, D::Error> {
        Self::deserialize(config)
    }
}

#[derive(Debug, Error)]
/// Errors that may occur when applying the [`LinearQuantizeCodec`].
pub enum LinearQuantizeCodecError {
    /// [`LinearQuantizeCodec`] cannot encode the provided dtype which differs
    /// from the configured dtype
    #[error("LinearQuantize cannot encode the provided dtype {provided} which differs from the configured dtype {configured}")]
    MismatchedEncodeDType {
        /// Dtype of the `configured` `dtype`
        configured: LinearQuantizeDType,
        /// Dtype of the `provided` array from which the data is to be encoded
        provided: AnyArrayDType,
    },
    /// [`LinearQuantizeCodec`] does not support non-finite (infinite or NaN) floating
    /// point data
    #[error("LinearQuantize does not support non-finite (infinite or NaN) floating point data")]
    NonFiniteData,
    /// [`LinearQuantizeCodec`] failed to encode the header
    #[error("LinearQuantize failed to encode the header")]
    HeaderEncodeFailed {
        /// Opaque source error
        source: LinearQuantizeHeaderError,
    },
    /// [`LinearQuantizeCodec`] can only decode one-dimensional arrays but
    /// received an array of a different shape
    #[error("LinearQuantize can only decode one-dimensional arrays but received an array of shape {shape:?}")]
    EncodedDataNotOneDimensional {
        /// The unexpected shape of the encoded array
        shape: Vec<usize>,
    },
    /// [`LinearQuantizeCodec`] failed to decode the header
    #[error("LinearQuantize failed to decode the header")]
    HeaderDecodeFailed {
        /// Opaque source error
        source: LinearQuantizeHeaderError,
    },
    /// [`LinearQuantizeCodec`] decoded an invalid array shape header which does
    /// not fit the decoded data
    #[error(
        "LinearQuantize decoded an invalid array shape header which does not fit the decoded data"
    )]
    DecodeInvalidShapeHeader {
        /// Source error
        #[from]
        source: ShapeError,
    },
    /// [`LinearQuantizeCodec`] cannot decode the provided dtype
    #[error("LinearQuantize cannot decode the provided dtype {dtype}")]
    InvalidEncodedDType {
        /// Dtype of the provided array from which the data is to be decoded
        dtype: AnyArrayDType,
    },
    /// [`LinearQuantizeCodec`] cannot decode the provided dtype which differs
    /// from the configured dtype
    #[error("LinearQuantize cannot decode the provided dtype {provided} which differs from the configured dtype {configured}")]
    MismatchedDecodeIntoDtype {
        /// Dtype of the `configured` `dtype`
        configured: LinearQuantizeDType,
        /// Dtype of the `provided` array into which the data is to be decoded
        provided: AnyArrayDType,
    },
    /// [`LinearQuantizeCodec`] cannot decode the decoded array into the provided
    /// array of a different shape
    #[error("LinearQuantize cannot decode the decoded array of shape {decoded:?} into the provided array of shape {provided:?}")]
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
pub struct LinearQuantizeHeaderError(postcard::Error);

/// Linear-quantize the elements in the `data` array using the `quantize`
/// closure.
///
/// # Errors
///
/// Errors with
/// - [`LinearQuantizeCodecError::NonFiniteData`] if any data element is non-
///   finite (infinite or NaN)
/// - [`LinearQuantizeCodecError::HeaderEncodeFailed`] if encoding the header
///   failed
pub fn quantize<T: Float, Q: Unsigned, S: Data<Elem = T>, D: Dimension>(
    data: &ArrayBase<S, D>,
    quantize: impl Fn(T) -> Q,
) -> Result<Vec<Q>, LinearQuantizeCodecError> {
    if !data.iter().copied().all(T::is_finite) {
        return Err(LinearQuantizeCodecError::NonFiniteData);
    }

    let minimum = data.iter().copied().reduce(T::minimum).unwrap_or(T::ZERO);
    let maximum = data.iter().copied().reduce(T::maximum).unwrap_or(T::ONE);

    let header = postcard::to_extend(
        &CompressionHeader {
            shape: Cow::Borrowed(data.shape()),
            minimum,
            maximum,
        },
        Vec::new(),
    )
    .map_err(|err| LinearQuantizeCodecError::HeaderEncodeFailed {
        source: LinearQuantizeHeaderError(err),
    })?;

    let mut encoded: Vec<Q> =
        vec![Q::ZERO; (header.len() + std::mem::size_of::<Q>() - 1) / std::mem::size_of::<Q>()];
    #[allow(unsafe_code)]
    // Safety: encoded is at least header.len() bytes long and propely aligned for Q
    unsafe {
        std::ptr::copy_nonoverlapping(header.as_ptr(), encoded.as_mut_ptr().cast(), header.len());
    }
    encoded.reserve(data.len());

    if maximum == minimum {
        encoded.resize(encoded.len() + data.len(), quantize(T::ZERO));
    } else {
        encoded.extend(
            data.iter()
                .map(|x| quantize((*x - minimum) / (maximum - minimum))),
        );
    }

    Ok(encoded)
}

/// Reconstruct the linear-quantized `encoded` array using the `floatify`
/// closure.
///
/// # Errors
///
/// Errors with
/// - [`LinearQuantizeCodecError::HeaderDecodeFailed`] if decoding the header
///   failed
pub fn reconstruct<T: Float, Q: Unsigned>(
    encoded: &[Q],
    floatify: impl Fn(Q) -> T,
) -> Result<ArrayD<T>, LinearQuantizeCodecError> {
    #[allow(unsafe_code)]
    // Safety: data is data.len()*size_of::<Q> bytes long and propely aligned for Q
    let (header, remaining) = postcard::take_from_bytes::<CompressionHeader<T>>(unsafe {
        std::slice::from_raw_parts(encoded.as_ptr().cast(), std::mem::size_of_val(encoded))
    })
    .map_err(|err| LinearQuantizeCodecError::HeaderDecodeFailed {
        source: LinearQuantizeHeaderError(err),
    })?;

    let encoded = encoded
        .get(encoded.len() - (remaining.len() / std::mem::size_of::<Q>())..)
        .unwrap_or(&[]);

    let decoded = encoded
        .iter()
        .map(|x| header.minimum + (floatify(*x) * (header.maximum - header.minimum)))
        .map(|x| x.clamp(header.minimum, header.maximum))
        .collect();

    let decoded = Array::from_shape_vec(&*header.shape, decoded)?;

    Ok(decoded)
}

#[allow(clippy::needless_pass_by_value)]
/// Reconstruct the linear-quantized `encoded` array using the `floatify`
/// closure into the `decoded` array.
///
/// # Errors
///
/// Errors with
/// - [`LinearQuantizeCodecError::HeaderDecodeFailed`] if decoding the header
///   failed
/// - [`LinearQuantizeCodecError::MismatchedDecodeIntoShape`] if the `decoded`
///   array is of the wrong shape
pub fn reconstruct_into<T: Float, Q: Unsigned>(
    encoded: &[Q],
    mut decoded: ArrayViewMutD<T>,
    floatify: impl Fn(Q) -> T,
) -> Result<(), LinearQuantizeCodecError> {
    #[allow(unsafe_code)]
    // Safety: data is data.len()*size_of::<Q> bytes long and propely aligned for Q
    let (header, remaining) = postcard::take_from_bytes::<CompressionHeader<T>>(unsafe {
        std::slice::from_raw_parts(encoded.as_ptr().cast(), std::mem::size_of_val(encoded))
    })
    .map_err(|err| LinearQuantizeCodecError::HeaderDecodeFailed {
        source: LinearQuantizeHeaderError(err),
    })?;

    let encoded = encoded
        .get(encoded.len() - (remaining.len() / std::mem::size_of::<Q>())..)
        .unwrap_or(&[]);

    if decoded.shape() != &*header.shape {
        return Err(LinearQuantizeCodecError::MismatchedDecodeIntoShape {
            decoded: header.shape.into_owned(),
            provided: decoded.shape().to_vec(),
        });
    }

    // iteration must occur in synchronised (standard) order
    for (e, d) in encoded.iter().zip(decoded.iter_mut()) {
        *d = (header.minimum + (floatify(*e) * (header.maximum - header.minimum)))
            .clamp(header.minimum, header.maximum);
    }

    Ok(())
}

/// Floating point types.
pub trait Float:
    Copy
    + Serialize
    + DeserializeOwned
    + std::ops::Sub<Self, Output = Self>
    + std::ops::Div<Self, Output = Self>
    + std::ops::Add<Self, Output = Self>
    + std::ops::Mul<Self, Output = Self>
    + PartialEq
{
    /// 0.0
    const ZERO: Self;
    /// 1.0
    const ONE: Self;

    /// Returns the minimum of the two numbers, ignoring NaN.
    #[must_use]
    fn minimum(self, other: Self) -> Self;

    /// Returns the maximum of the two numbers, ignoring NaN.
    #[must_use]
    fn maximum(self, other: Self) -> Self;

    /// Restrict a value to a certain interval unless it is NaN.
    #[must_use]
    fn clamp(self, min: Self, max: Self) -> Self;

    /// Returns `exp2(bits) - 1.0`
    fn scale_for_bits(bits: u8) -> Self;

    /// Returns `true` if this number is neither infinite nor NaN.
    fn is_finite(self) -> bool;
}

impl Float for f32 {
    const ONE: Self = 1.0;
    const ZERO: Self = 0.0;

    fn minimum(self, other: Self) -> Self {
        Self::min(self, other)
    }

    fn maximum(self, other: Self) -> Self {
        Self::max(self, other)
    }

    fn clamp(self, min: Self, max: Self) -> Self {
        Self::clamp(self, min, max)
    }

    fn scale_for_bits(bits: u8) -> Self {
        Self::from(bits).exp2() - 1.0
    }

    fn is_finite(self) -> bool {
        self.is_finite()
    }
}

impl Float for f64 {
    const ONE: Self = 1.0;
    const ZERO: Self = 0.0;

    fn minimum(self, other: Self) -> Self {
        Self::min(self, other)
    }

    fn maximum(self, other: Self) -> Self {
        Self::max(self, other)
    }

    fn clamp(self, min: Self, max: Self) -> Self {
        Self::clamp(self, min, max)
    }

    fn scale_for_bits(bits: u8) -> Self {
        Self::from(bits).exp2() - 1.0
    }

    fn is_finite(self) -> bool {
        self.is_finite()
    }
}

/// Unsigned binary types.
pub trait Unsigned: Copy {
    /// `0x0`
    const ZERO: Self;
}

impl Unsigned for u8 {
    const ZERO: Self = 0;
}

impl Unsigned for u16 {
    const ZERO: Self = 0;
}

impl Unsigned for u32 {
    const ZERO: Self = 0;
}

impl Unsigned for u64 {
    const ZERO: Self = 0;
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
struct CompressionHeader<'a, T: Float> {
    #[serde(borrow)]
    shape: Cow<'a, [usize]>,
    minimum: T,
    maximum: T,
}

#[cfg(test)]
mod tests {
    use ndarray::CowArray;

    use super::*;

    #[test]
    fn exact_roundtrip_f32() -> Result<(), LinearQuantizeCodecError> {
        for bits in 1..=16 {
            let codec = LinearQuantizeCodec {
                dtype: LinearQuantizeDType::F32,
                #[allow(unsafe_code)]
                bits: unsafe { std::mem::transmute::<u8, LinearQuantizeBins>(bits) },
            };

            let mut data: Vec<f32> = (0..(u16::MAX >> (16 - bits)))
                .step_by(1 << (bits.max(8) - 8))
                .map(f32::from)
                .collect();
            data.push(f32::from(u16::MAX >> (16 - bits)));

            let encoded = codec.encode(AnyCowArray::F32(CowArray::from(&data).into_dyn()))?;
            let decoded = codec.decode(encoded.cow())?;

            let AnyArray::F32(decoded) = decoded else {
                return Err(LinearQuantizeCodecError::MismatchedDecodeIntoDtype {
                    configured: LinearQuantizeDType::F32,
                    provided: decoded.dtype(),
                });
            };

            for (o, d) in data.iter().zip(decoded.iter()) {
                assert_eq!(o.to_bits(), d.to_bits());
            }
        }

        Ok(())
    }

    #[test]
    fn almost_roundtrip_f32() -> Result<(), LinearQuantizeCodecError> {
        for bits in 17..=64 {
            let codec = LinearQuantizeCodec {
                dtype: LinearQuantizeDType::F32,
                #[allow(unsafe_code)]
                bits: unsafe { std::mem::transmute::<u8, LinearQuantizeBins>(bits) },
            };

            #[allow(clippy::cast_precision_loss)]
            let mut data: Vec<f32> = (0..(u64::MAX >> (64 - bits)))
                .step_by(1 << (bits.max(8) - 8))
                .map(|x| x as f32)
                .collect();
            #[allow(clippy::cast_precision_loss)]
            data.push((u64::MAX >> (64 - bits)) as f32);

            let encoded = codec.encode(AnyCowArray::F32(CowArray::from(&data).into_dyn()))?;
            let decoded = codec.decode(encoded.cow())?;

            let AnyArray::F32(decoded) = decoded else {
                return Err(LinearQuantizeCodecError::MismatchedDecodeIntoDtype {
                    configured: LinearQuantizeDType::F32,
                    provided: decoded.dtype(),
                });
            };

            for (o, d) in data.iter().zip(decoded.iter()) {
                // FIXME: there seem to be some rounding errors
                assert!((o - d).abs() <= 1.0);
            }
        }

        Ok(())
    }

    #[test]
    fn exact_roundtrip_f64() -> Result<(), LinearQuantizeCodecError> {
        for bits in 1..=32 {
            let codec = LinearQuantizeCodec {
                dtype: LinearQuantizeDType::F64,
                #[allow(unsafe_code)]
                bits: unsafe { std::mem::transmute::<u8, LinearQuantizeBins>(bits) },
            };

            let mut data: Vec<f64> = (0..(u32::MAX >> (32 - bits)))
                .step_by(1 << (bits.max(8) - 8))
                .map(f64::from)
                .collect();
            data.push(f64::from(u32::MAX >> (32 - bits)));

            let encoded = codec.encode(AnyCowArray::F64(CowArray::from(&data).into_dyn()))?;
            let decoded = codec.decode(encoded.cow())?;

            let AnyArray::F64(decoded) = decoded else {
                return Err(LinearQuantizeCodecError::MismatchedDecodeIntoDtype {
                    configured: LinearQuantizeDType::F64,
                    provided: decoded.dtype(),
                });
            };

            for (o, d) in data.iter().zip(decoded.iter()) {
                assert_eq!(o.to_bits(), d.to_bits());
            }
        }

        Ok(())
    }

    #[test]
    fn almost_roundtrip_f64() -> Result<(), LinearQuantizeCodecError> {
        for bits in 33..=64 {
            let codec = LinearQuantizeCodec {
                dtype: LinearQuantizeDType::F64,
                #[allow(unsafe_code)]
                bits: unsafe { std::mem::transmute::<u8, LinearQuantizeBins>(bits) },
            };

            #[allow(clippy::cast_precision_loss)]
            let mut data: Vec<f64> = (0..(u64::MAX >> (64 - bits)))
                .step_by(1 << (bits.max(8) - 8))
                .map(|x| x as f64)
                .collect();
            #[allow(clippy::cast_precision_loss)]
            data.push((u64::MAX >> (64 - bits)) as f64);

            let encoded = codec.encode(AnyCowArray::F64(CowArray::from(&data).into_dyn()))?;
            let decoded = codec.decode(encoded.cow())?;

            let AnyArray::F64(decoded) = decoded else {
                return Err(LinearQuantizeCodecError::MismatchedDecodeIntoDtype {
                    configured: LinearQuantizeDType::F64,
                    provided: decoded.dtype(),
                });
            };

            for (o, d) in data.iter().zip(decoded.iter()) {
                // FIXME: there seem to be some rounding errors
                assert!((o - d).abs() < 2.0);
            }
        }

        Ok(())
    }

    #[test]
    fn const_data_roundtrip() -> Result<(), LinearQuantizeCodecError> {
        for bits in 1..=64 {
            let data = [42.0, 42.0, 42.0, 42.0];

            let codec = LinearQuantizeCodec {
                dtype: LinearQuantizeDType::F64,
                #[allow(unsafe_code)]
                bits: unsafe { std::mem::transmute::<u8, LinearQuantizeBins>(bits) },
            };

            let encoded = codec.encode(AnyCowArray::F64(CowArray::from(&data).into_dyn()))?;
            let decoded = codec.decode(encoded.cow())?;

            let AnyArray::F64(decoded) = decoded else {
                return Err(LinearQuantizeCodecError::MismatchedDecodeIntoDtype {
                    configured: LinearQuantizeDType::F64,
                    provided: decoded.dtype(),
                });
            };

            for (o, d) in data.iter().zip(decoded.iter()) {
                assert_eq!(o.to_bits(), d.to_bits());
            }
        }

        Ok(())
    }
}
