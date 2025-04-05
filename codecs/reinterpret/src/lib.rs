//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.82.0-blue
//! [repo]: https://github.com/juntyr/numcodecs-rs
//!
//! [Latest Version]: https://img.shields.io/crates/v/numcodecs-reinterpret
//! [crates.io]: https://crates.io/crates/numcodecs-reinterpret
//!
//! [Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs-reinterpret
//! [docs.rs]: https://docs.rs/numcodecs-reinterpret/
//!
//! [Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/numcodecs-rs/numcodecs_reinterpret
//!
//! Binary reinterpret codec implementation for the [`numcodecs`] API.

use ndarray::{Array, ArrayBase, ArrayView, Data, DataMut, Dimension, ViewRepr};
use numcodecs::{
    AnyArray, AnyArrayAssignError, AnyArrayDType, AnyArrayView, AnyArrayViewMut, AnyCowArray,
    ArrayDType, Codec, StaticCodec, StaticCodecConfig, StaticCodecVersion,
};
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

#[derive(Clone, JsonSchema)]
#[serde(deny_unknown_fields)]
/// Codec to reinterpret data between different compatible types.
///
/// Note that no conversion happens, only the meaning of the bits changes.
///
/// Reinterpreting to bytes, or to a same-sized unsigned integer type, or
/// without the changing the dtype are supported.
pub struct ReinterpretCodec {
    /// Dtype of the encoded data.
    encode_dtype: AnyArrayDType,
    /// Dtype of the decoded data
    decode_dtype: AnyArrayDType,
    /// The codec's encoding format version. Do not provide this parameter explicitly.
    #[serde(default)]
    _version: StaticCodecVersion<1, 0, 0>,
}

impl ReinterpretCodec {
    /// Try to create a [`ReinterpretCodec`] that reinterprets the input data
    /// from `decode_dtype` to `encode_dtype` on encoding, and from
    /// `encode_dtype` back to `decode_dtype` on decoding.
    ///
    /// # Errors
    ///
    /// Errors with [`ReinterpretCodecError::InvalidReinterpret`] if
    /// `encode_dtype` and `decode_dtype` are incompatible.
    pub fn try_new(
        encode_dtype: AnyArrayDType,
        decode_dtype: AnyArrayDType,
    ) -> Result<Self, ReinterpretCodecError> {
        #[expect(clippy::match_same_arms)]
        match (decode_dtype, encode_dtype) {
            // performing no conversion always works
            (ty_a, ty_b) if ty_a == ty_b => (),
            // converting to bytes always works
            (_, AnyArrayDType::U8) => (),
            // converting from signed / floating to same-size binary always works
            (AnyArrayDType::I16, AnyArrayDType::U16)
            | (AnyArrayDType::I32 | AnyArrayDType::F32, AnyArrayDType::U32)
            | (AnyArrayDType::I64 | AnyArrayDType::F64, AnyArrayDType::U64) => (),
            (decode_dtype, encode_dtype) => {
                return Err(ReinterpretCodecError::InvalidReinterpret {
                    decode_dtype,
                    encode_dtype,
                })
            }
        }

        Ok(Self {
            encode_dtype,
            decode_dtype,
            _version: StaticCodecVersion,
        })
    }

    #[must_use]
    /// Create a [`ReinterpretCodec`] that does not change the `dtype`.
    pub const fn passthrough(dtype: AnyArrayDType) -> Self {
        Self {
            encode_dtype: dtype,
            decode_dtype: dtype,
            _version: StaticCodecVersion,
        }
    }

    #[must_use]
    /// Create a [`ReinterpretCodec`] that reinterprets `dtype` as
    /// [bytes][`AnyArrayDType::U8`].
    pub const fn to_bytes(dtype: AnyArrayDType) -> Self {
        Self {
            encode_dtype: AnyArrayDType::U8,
            decode_dtype: dtype,
            _version: StaticCodecVersion,
        }
    }

    #[must_use]
    /// Create a  [`ReinterpretCodec`] that reinterprets `dtype` as its
    /// [binary][`AnyArrayDType::to_binary`] equivalent.
    pub const fn to_binary(dtype: AnyArrayDType) -> Self {
        Self {
            encode_dtype: dtype.to_binary(),
            decode_dtype: dtype,
            _version: StaticCodecVersion,
        }
    }
}

impl Codec for ReinterpretCodec {
    type Error = ReinterpretCodecError;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        if data.dtype() != self.decode_dtype {
            return Err(ReinterpretCodecError::MismatchedEncodeDType {
                configured: self.decode_dtype,
                provided: data.dtype(),
            });
        }

        let encoded = match (data, self.encode_dtype) {
            (data, dtype) if data.dtype() == dtype => data.into_owned(),
            (data, AnyArrayDType::U8) => {
                let mut shape = data.shape().to_vec();
                if let Some(last) = shape.last_mut() {
                    *last *= data.dtype().size();
                }
                #[expect(unsafe_code)]
                // Safety: the shape is extended to match the expansion into bytes
                let encoded =
                    unsafe { Array::from_shape_vec_unchecked(shape, data.as_bytes().into_owned()) };
                AnyArray::U8(encoded)
            }
            (AnyCowArray::I16(data), AnyArrayDType::U16) => {
                AnyArray::U16(reinterpret_array(data, |x| {
                    u16::from_ne_bytes(x.to_ne_bytes())
                }))
            }
            (AnyCowArray::I32(data), AnyArrayDType::U32) => {
                AnyArray::U32(reinterpret_array(data, |x| {
                    u32::from_ne_bytes(x.to_ne_bytes())
                }))
            }
            (AnyCowArray::F32(data), AnyArrayDType::U32) => {
                AnyArray::U32(reinterpret_array(data, f32::to_bits))
            }
            (AnyCowArray::I64(data), AnyArrayDType::U64) => {
                AnyArray::U64(reinterpret_array(data, |x| {
                    u64::from_ne_bytes(x.to_ne_bytes())
                }))
            }
            (AnyCowArray::F64(data), AnyArrayDType::U64) => {
                AnyArray::U64(reinterpret_array(data, f64::to_bits))
            }
            (data, dtype) => {
                return Err(ReinterpretCodecError::InvalidReinterpret {
                    decode_dtype: data.dtype(),
                    encode_dtype: dtype,
                });
            }
        };

        Ok(encoded)
    }

    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
        if encoded.dtype() != self.encode_dtype {
            return Err(ReinterpretCodecError::MismatchedDecodeDType {
                configured: self.encode_dtype,
                provided: encoded.dtype(),
            });
        }

        let decoded = match (encoded, self.decode_dtype) {
            (encoded, dtype) if encoded.dtype() == dtype => encoded.into_owned(),
            (AnyCowArray::U8(encoded), dtype) => {
                let mut shape = encoded.shape().to_vec();

                if (encoded.len() % dtype.size()) != 0 {
                    return Err(ReinterpretCodecError::InvalidEncodedShape { shape, dtype });
                }

                if let Some(last) = shape.last_mut() {
                    *last /= dtype.size();
                }

                let (decoded, ()) = AnyArray::with_zeros_bytes(dtype, &shape, |bytes| {
                    bytes.copy_from_slice(&AnyCowArray::U8(encoded).as_bytes());
                });

                decoded
            }
            (AnyCowArray::U16(encoded), AnyArrayDType::I16) => {
                AnyArray::I16(reinterpret_array(encoded, |x| {
                    i16::from_ne_bytes(x.to_ne_bytes())
                }))
            }
            (AnyCowArray::U32(encoded), AnyArrayDType::I32) => {
                AnyArray::I32(reinterpret_array(encoded, |x| {
                    i32::from_ne_bytes(x.to_ne_bytes())
                }))
            }
            (AnyCowArray::U32(encoded), AnyArrayDType::F32) => {
                AnyArray::F32(reinterpret_array(encoded, f32::from_bits))
            }
            (AnyCowArray::U64(encoded), AnyArrayDType::U64) => {
                AnyArray::I64(reinterpret_array(encoded, |x| {
                    i64::from_ne_bytes(x.to_ne_bytes())
                }))
            }
            (AnyCowArray::U64(encoded), AnyArrayDType::F64) => {
                AnyArray::F64(reinterpret_array(encoded, f64::from_bits))
            }
            (encoded, dtype) => {
                return Err(ReinterpretCodecError::InvalidReinterpret {
                    decode_dtype: dtype,
                    encode_dtype: encoded.dtype(),
                });
            }
        };

        Ok(decoded)
    }

    fn decode_into(
        &self,
        encoded: AnyArrayView,
        mut decoded: AnyArrayViewMut,
    ) -> Result<(), Self::Error> {
        if encoded.dtype() != self.encode_dtype {
            return Err(ReinterpretCodecError::MismatchedDecodeDType {
                configured: self.encode_dtype,
                provided: encoded.dtype(),
            });
        }

        match (encoded, self.decode_dtype) {
            (encoded, dtype) if encoded.dtype() == dtype => Ok(decoded.assign(&encoded)?),
            (AnyArrayView::U8(encoded), dtype) => {
                if decoded.dtype() != dtype {
                    return Err(ReinterpretCodecError::MismatchedDecodeIntoArray {
                        source: AnyArrayAssignError::DTypeMismatch {
                            src: dtype,
                            dst: decoded.dtype(),
                        },
                    });
                }

                let mut shape = encoded.shape().to_vec();

                if (encoded.len() % dtype.size()) != 0 {
                    return Err(ReinterpretCodecError::InvalidEncodedShape { shape, dtype });
                }

                if let Some(last) = shape.last_mut() {
                    *last /= dtype.size();
                }

                if decoded.shape() != shape {
                    return Err(ReinterpretCodecError::MismatchedDecodeIntoArray {
                        source: AnyArrayAssignError::ShapeMismatch {
                            src: shape,
                            dst: decoded.shape().to_vec(),
                        },
                    });
                }

                let () = decoded.with_bytes_mut(|bytes| {
                    bytes.copy_from_slice(&AnyArrayView::U8(encoded).as_bytes());
                });

                Ok(())
            }
            (AnyArrayView::U16(encoded), AnyArrayDType::I16) => {
                reinterpret_array_into(encoded, |x| i16::from_ne_bytes(x.to_ne_bytes()), decoded)
            }
            (AnyArrayView::U32(encoded), AnyArrayDType::I32) => {
                reinterpret_array_into(encoded, |x| i32::from_ne_bytes(x.to_ne_bytes()), decoded)
            }
            (AnyArrayView::U32(encoded), AnyArrayDType::F32) => {
                reinterpret_array_into(encoded, f32::from_bits, decoded)
            }
            (AnyArrayView::U64(encoded), AnyArrayDType::U64) => {
                reinterpret_array_into(encoded, |x| i64::from_ne_bytes(x.to_ne_bytes()), decoded)
            }
            (AnyArrayView::U64(encoded), AnyArrayDType::F64) => {
                reinterpret_array_into(encoded, f64::from_bits, decoded)
            }
            (encoded, dtype) => Err(ReinterpretCodecError::InvalidReinterpret {
                decode_dtype: dtype,
                encode_dtype: encoded.dtype(),
            }),
        }?;

        Ok(())
    }
}

impl StaticCodec for ReinterpretCodec {
    const CODEC_ID: &'static str = "reinterpret.rs";

    type Config<'de> = Self;

    fn from_config(config: Self::Config<'_>) -> Self {
        config
    }

    fn get_config(&self) -> StaticCodecConfig<Self> {
        StaticCodecConfig::from(self)
    }
}

impl Serialize for ReinterpretCodec {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        ReinterpretCodecConfig {
            encode_dtype: self.encode_dtype,
            decode_dtype: self.decode_dtype,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ReinterpretCodec {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let config = ReinterpretCodecConfig::deserialize(deserializer)?;

        Self::try_new(config.encode_dtype, config.decode_dtype).map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename = "ReinterpretCodec")]
struct ReinterpretCodecConfig {
    encode_dtype: AnyArrayDType,
    decode_dtype: AnyArrayDType,
}

#[derive(Debug, Error)]
/// Errors that may occur when applying the [`ReinterpretCodec`].
pub enum ReinterpretCodecError {
    /// [`ReinterpretCodec`] cannot cannot bitcast the `decode_dtype` as
    /// `encode_dtype`
    #[error("Reinterpret cannot bitcast {decode_dtype} as {encode_dtype}")]
    InvalidReinterpret {
        /// Dtype of the configured `decode_dtype`
        decode_dtype: AnyArrayDType,
        /// Dtype of the configured `encode_dtype`
        encode_dtype: AnyArrayDType,
    },
    /// [`ReinterpretCodec`] cannot encode the provided dtype which differs
    /// from the configured dtype
    #[error("Reinterpret cannot encode the provided dtype {provided} which differs from the configured dtype {configured}")]
    MismatchedEncodeDType {
        /// Dtype of the `configured` `decode_dtype`
        configured: AnyArrayDType,
        /// Dtype of the `provided` array from which the data is to be encoded
        provided: AnyArrayDType,
    },
    /// [`ReinterpretCodec`] cannot decode the provided dtype which differs
    /// from the configured dtype
    #[error("Reinterpret cannot decode the provided dtype {provided} which differs from the configured dtype {configured}")]
    MismatchedDecodeDType {
        /// Dtype of the `configured` `encode_dtype`
        configured: AnyArrayDType,
        /// Dtype of the `provided` array from which the data is to be decoded
        provided: AnyArrayDType,
    },
    /// [`ReinterpretCodec`] cannot decode a byte array with `shape` into an array of `dtype`s
    #[error(
        "Reinterpret cannot decode a byte array of shape {shape:?} into an array of {dtype}-s"
    )]
    InvalidEncodedShape {
        /// Shape of the encoded array
        shape: Vec<usize>,
        /// Dtype of the array into which the encoded data is to be decoded
        dtype: AnyArrayDType,
    },
    /// [`ReinterpretCodec`] cannot decode into the provided array
    #[error("Reinterpret cannot decode into the provided array")]
    MismatchedDecodeIntoArray {
        /// The source of the error
        #[from]
        source: AnyArrayAssignError,
    },
}

/// Reinterpret the data elements of the `array` using the provided `reinterpret`
/// closure. The shape of the data is preserved.
#[inline]
pub fn reinterpret_array<T: Copy, U, S: Data<Elem = T>, D: Dimension>(
    array: ArrayBase<S, D>,
    reinterpret: impl Fn(T) -> U,
) -> Array<U, D> {
    let array = array.into_owned();
    let (shape, data) = (array.raw_dim(), array.into_raw_vec_and_offset().0);

    let data = data.into_iter().map(reinterpret).collect();

    #[expect(unsafe_code)]
    // Safety: we have preserved the shape, which comes from a valid array
    let array = unsafe { Array::from_shape_vec_unchecked(shape, data) };

    array
}

#[expect(clippy::needless_pass_by_value)]
/// Reinterpret the data elements of the `encoded` array using the provided
/// `reinterpret` closure into the `decoded` array.
///
/// # Errors
///
/// Errors with
/// - [`ReinterpretCodecError::MismatchedDecodeIntoArray`] if `decoded` does not
///   contain an array with elements of type `U` or its shape does not match the
///   `encoded` array's shape
#[inline]
pub fn reinterpret_array_into<'a, T: Copy, U: ArrayDType, D: Dimension>(
    encoded: ArrayView<T, D>,
    reinterpret: impl Fn(T) -> U,
    mut decoded: AnyArrayViewMut<'a>,
) -> Result<(), ReinterpretCodecError>
where
    U::RawData<ViewRepr<&'a mut ()>>: DataMut,
{
    let Some(decoded) = decoded.as_typed_mut::<U>() else {
        return Err(ReinterpretCodecError::MismatchedDecodeIntoArray {
            source: AnyArrayAssignError::DTypeMismatch {
                src: U::DTYPE,
                dst: decoded.dtype(),
            },
        });
    };

    if encoded.shape() != decoded.shape() {
        return Err(ReinterpretCodecError::MismatchedDecodeIntoArray {
            source: AnyArrayAssignError::ShapeMismatch {
                src: encoded.shape().to_vec(),
                dst: decoded.shape().to_vec(),
            },
        });
    }

    // iterate over the elements in standard order
    for (e, d) in encoded.iter().zip(decoded.iter_mut()) {
        *d = reinterpret(*e);
    }

    Ok(())
}
