//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.76.0-blue
//! [repo]: https://github.com/juntyr/numcodecs-rs
//!
//! [Latest Version]: https://img.shields.io/crates/v/numcodecs-log
//! [crates.io]: https://crates.io/crates/numcodecs-log
//!
//! [Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs-log
//! [docs.rs]: https://docs.rs/numcodecs-log/
//!
//! [Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/numcodecs-rs/numcodecs_log
//!
//! `(x-o) * s` codec implementation for the [`numcodecs`] API.

use ndarray::{Array, ArrayBase, ArrayView, ArrayViewMut, Data, Dimension};
use numcodecs::{
    AnyArray, AnyArrayAssignError, AnyArrayDType, AnyArrayView, AnyArrayViewMut, AnyCowArray,
    Codec, StaticCodec, StaticCodecConfig,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
/// Fixed offset-scale codec which calculates `c = (x-o) / s` on encoding and
/// `d = (c*s) + o` on decoding.
///
/// - Setting `o = mean(x)` and `s = std(x)` normalizes that data.
/// - Setting `o = min(x)` and `s = max(x) - min(x)` standardizes the data.
///
/// The codec only supports floating point numbers.
pub struct FixedOffsetScaleCodec {
    /// The offset of the data.
    pub offset: f64,
    /// The scale of the data.
    pub scale: f64,
}

impl Codec for FixedOffsetScaleCodec {
    type Error = FixedOffsetScaleCodecError;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        match data {
            #[allow(clippy::cast_possible_truncation)]
            AnyCowArray::F32(data) => Ok(AnyArray::F32(scale(
                data,
                self.offset as f32,
                self.scale as f32,
            ))),
            AnyCowArray::F64(data) => Ok(AnyArray::F64(scale(data, self.offset, self.scale))),
            encoded => Err(FixedOffsetScaleCodecError::UnsupportedDtype(
                encoded.dtype(),
            )),
        }
    }

    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
        match encoded {
            #[allow(clippy::cast_possible_truncation)]
            AnyCowArray::F32(encoded) => Ok(AnyArray::F32(unscale(
                encoded,
                self.offset as f32,
                self.scale as f32,
            ))),
            AnyCowArray::F64(encoded) => {
                Ok(AnyArray::F64(unscale(encoded, self.offset, self.scale)))
            }
            encoded => Err(FixedOffsetScaleCodecError::UnsupportedDtype(
                encoded.dtype(),
            )),
        }
    }

    fn decode_into(
        &self,
        encoded: AnyArrayView,
        decoded: AnyArrayViewMut,
    ) -> Result<(), Self::Error> {
        match (encoded, decoded) {
            #[allow(clippy::cast_possible_truncation)]
            (AnyArrayView::F32(encoded), AnyArrayViewMut::F32(decoded)) => {
                unscale_into(encoded, decoded, self.offset as f32, self.scale as f32)
            }
            (AnyArrayView::F64(encoded), AnyArrayViewMut::F64(decoded)) => {
                unscale_into(encoded, decoded, self.offset, self.scale)
            }
            (encoded @ (AnyArrayView::F32(_) | AnyArrayView::F64(_)), decoded) => {
                Err(FixedOffsetScaleCodecError::MismatchedDecodeIntoArray {
                    source: AnyArrayAssignError::DTypeMismatch {
                        src: encoded.dtype(),
                        dst: decoded.dtype(),
                    },
                })
            }
            (encoded, _decoded) => Err(FixedOffsetScaleCodecError::UnsupportedDtype(
                encoded.dtype(),
            )),
        }
    }
}

impl StaticCodec for FixedOffsetScaleCodec {
    const CODEC_ID: &'static str = "fixed-offset-scale";

    type Config<'de> = Self;

    fn from_config(config: Self::Config<'_>) -> Self {
        config
    }

    fn get_config(&self) -> StaticCodecConfig<Self> {
        StaticCodecConfig::from(self)
    }
}

#[derive(Debug, Error)]
/// Errors that may occur when applying the [`FixedOffsetScaleCodec`].
pub enum FixedOffsetScaleCodecError {
    /// [`FixedOffsetScaleCodec`] does not support the dtype
    #[error("FixedOffsetScale does not support the dtype {0}")]
    UnsupportedDtype(AnyArrayDType),
    /// [`FixedOffsetScaleCodec`] cannot decode into the provided array
    #[error("FixedOffsetScale cannot decode into the provided array")]
    MismatchedDecodeIntoArray {
        /// The source of the error
        #[from]
        source: AnyArrayAssignError,
    },
}

/// Compute `(x-o) / s` over the elements of the input `data` array.
pub fn scale<T: Float, S: Data<Elem = T>, D: Dimension>(
    data: ArrayBase<S, D>,
    offset: T,
    scale: T,
) -> Array<T, D> {
    let negative_offset = offset.negate();
    let inverse_scale = scale.recip();

    let mut data = data.into_owned();
    data.mapv_inplace(|x| x.add_mul(negative_offset, inverse_scale));

    data
}

/// Compute `(x*s) + o` over the elements of the input `data` array.
pub fn unscale<T: Float, S: Data<Elem = T>, D: Dimension>(
    data: ArrayBase<S, D>,
    offset: T,
    scale: T,
) -> Array<T, D> {
    let mut data = data.into_owned();
    data.mapv_inplace(|x| x.mul_add(offset, scale));

    data
}

#[allow(clippy::needless_pass_by_value)]
/// Compute `(x*s) + o` over the elements of the input `data` array and write
/// them into the `out`put array.
///
/// # Errors
///
/// Errors with
/// - [`FixedOffsetScaleCodecError::MismatchedDecodeIntoArray`] if the `data`
///   array's shape does not match the `out`put array's shape
pub fn unscale_into<T: Float, D: Dimension>(
    data: ArrayView<T, D>,
    mut out: ArrayViewMut<T, D>,
    offset: T,
    scale: T,
) -> Result<(), FixedOffsetScaleCodecError> {
    if data.shape() != out.shape() {
        return Err(FixedOffsetScaleCodecError::MismatchedDecodeIntoArray {
            source: AnyArrayAssignError::ShapeMismatch {
                src: data.shape().to_vec(),
                dst: out.shape().to_vec(),
            },
        });
    }

    // iteration must occur in synchronised (standard) order
    for (d, o) in data.iter().zip(out.iter_mut()) {
        *o = (*d).mul_add(offset, scale);
    }

    Ok(())
}

/// Floating point types.
pub trait Float: Copy {
    /// Returns `-self`.
    #[must_use]
    fn negate(self) -> Self;

    /// Returns the reciprocal (inverse) of a number, `1/self`.
    #[must_use]
    fn recip(self) -> Self;

    /// Returns `(self + offset) * scale`.
    #[must_use]
    fn add_mul(self, offset: Self, scale: Self) -> Self;

    /// Returns `(self * scale) + offset`.
    #[must_use]
    fn mul_add(self, offset: Self, scale: Self) -> Self;
}

impl Float for f32 {
    fn negate(self) -> Self {
        -self
    }

    fn recip(self) -> Self {
        self.recip()
    }

    fn add_mul(self, offset: Self, scale: Self) -> Self {
        (self + offset) * scale
    }

    fn mul_add(self, offset: Self, scale: Self) -> Self {
        self.mul_add(scale, offset)
    }
}

impl Float for f64 {
    fn negate(self) -> Self {
        -self
    }

    fn recip(self) -> Self {
        self.recip()
    }

    fn add_mul(self, offset: Self, scale: Self) -> Self {
        (self + offset) * scale
    }

    fn mul_add(self, offset: Self, scale: Self) -> Self {
        self.mul_add(scale, offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity() {
        let data = (0..1000).map(|x| x as f64).collect::<Vec<_>>();
        let data = Array::from_vec(data);

        let encoded = scale(data.view(), 0.0, 1.0);

        for (r, e) in data.iter().zip(encoded.iter()) {
            assert_eq!((*r).to_bits(), (*e).to_bits());
        }

        let decoded = unscale(encoded, 0.0, 1.0);

        for (r, d) in data.iter().zip(decoded.iter()) {
            assert_eq!((*r).to_bits(), (*d).to_bits());
        }
    }

    #[test]
    fn roundtrip() -> Result<(), FixedOffsetScaleCodecError> {
        let data = (0..1000).map(|x| x as f64).collect::<Vec<_>>();
        let data = Array::from_vec(data);

        let encoded = scale(data.view(), 512.0, 64.0);

        for (r, e) in data.iter().zip(encoded.iter()) {
            assert_eq!((((*r) - 512.0) / 64.0).to_bits(), (*e).to_bits());
        }

        let decoded = unscale(encoded, 512.0, 64.0);

        for (r, d) in data.iter().zip(decoded.iter()) {
            assert_eq!((*r).to_bits(), (*d).to_bits());
        }

        Ok(())
    }
}