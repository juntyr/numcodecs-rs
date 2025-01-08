//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.82.0-blue
//! [repo]: https://github.com/juntyr/numcodecs-rs
//!
//! [Latest Version]: https://img.shields.io/crates/v/numcodecs-asinh
//! [crates.io]: https://crates.io/crates/numcodecs-asinh
//!
//! [Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs-asinh
//! [docs.rs]: https://docs.rs/numcodecs-asinh/
//!
//! [Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/numcodecs-rs/numcodecs_asinh
//!
//! `asinh(x)` codec implementation for the [`numcodecs`] API.

use ndarray::{Array, ArrayBase, ArrayView, ArrayViewMut, Data, Dimension, Zip};
use num_traits::{Float, Signed};
use numcodecs::{
    AnyArray, AnyArrayAssignError, AnyArrayDType, AnyArrayView, AnyArrayViewMut, AnyCowArray,
    Codec, StaticCodec, StaticCodecConfig,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
/// Asinh codec, which applies a quasi-logarithmic transformation on encoding.
///
/// For values close to zero that are within the codec's `linear_width`, the
/// transform is close to linear. For values of larger magnitudes, the
/// transform is asymptotically logarithmic. Unlike a logarithmic transform,
/// this codec supports all finite values, including negative values and zero.
///
/// In detail, the codec calculates `c = asinh(x/w) * w` on encoding and
/// `d = sinh(c/w) * w` on decoding, where `w` is the codec's `linear_width`.
///
/// The codec only supports finite floating point numbers.
pub struct AsinhCodec {
    /// The width of the close-to-zero input value range where the transform is
    /// nearly linear
    linear_width: f64,
}

impl Codec for AsinhCodec {
    type Error = AsinhCodecError;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        match data {
            #[allow(clippy::cast_possible_truncation)]
            AnyCowArray::F32(data) => Ok(AnyArray::F32(asinh(data, self.linear_width as f32)?)),
            AnyCowArray::F64(data) => Ok(AnyArray::F64(asinh(data, self.linear_width)?)),
            encoded => Err(AsinhCodecError::UnsupportedDtype(encoded.dtype())),
        }
    }

    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
        match encoded {
            #[allow(clippy::cast_possible_truncation)]
            AnyCowArray::F32(encoded) => {
                Ok(AnyArray::F32(sinh(encoded, self.linear_width as f32)?))
            }
            AnyCowArray::F64(encoded) => Ok(AnyArray::F64(sinh(encoded, self.linear_width)?)),
            encoded => Err(AsinhCodecError::UnsupportedDtype(encoded.dtype())),
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
                sinh_into(encoded, decoded, self.linear_width as f32)
            }
            (AnyArrayView::F64(encoded), AnyArrayViewMut::F64(decoded)) => {
                sinh_into(encoded, decoded, self.linear_width)
            }
            (encoded @ (AnyArrayView::F32(_) | AnyArrayView::F64(_)), decoded) => {
                Err(AsinhCodecError::MismatchedDecodeIntoArray {
                    source: AnyArrayAssignError::DTypeMismatch {
                        src: encoded.dtype(),
                        dst: decoded.dtype(),
                    },
                })
            }
            (encoded, _decoded) => Err(AsinhCodecError::UnsupportedDtype(encoded.dtype())),
        }
    }
}

impl StaticCodec for AsinhCodec {
    const CODEC_ID: &'static str = "asinh";

    type Config<'de> = Self;

    fn from_config(config: Self::Config<'_>) -> Self {
        config
    }

    fn get_config(&self) -> StaticCodecConfig<Self> {
        StaticCodecConfig::from(self)
    }
}

#[derive(Debug, Error)]
/// Errors that may occur when applying the [`AsinhCodec`].
pub enum AsinhCodecError {
    /// [`AsinhCodec`] does not support the dtype
    #[error("Asinh does not support the dtype {0}")]
    UnsupportedDtype(AnyArrayDType),
    /// [`AsinhCodec`] does not support non-finite (infinite or NaN) floating
    /// point data
    #[error("Asinh does not support non-finite (infinite or NaN) floating point data")]
    NonFiniteData,
    /// [`AsinhCodec`] cannot decode into the provided array
    #[error("Asinh cannot decode into the provided array")]
    MismatchedDecodeIntoArray {
        /// The source of the error
        #[from]
        source: AnyArrayAssignError,
    },
}

/// Compute `asinh(x/w) * w` over the elements of the input `data` array.
///
/// # Errors
///
/// Errors with
/// - [`AsinhCodecError::NonFiniteData`] if any data element is non-finite
///   (infinite or NaN)
pub fn asinh<T: Float + Signed, S: Data<Elem = T>, D: Dimension>(
    data: ArrayBase<S, D>,
    linear_width: T,
) -> Result<Array<T, D>, AsinhCodecError> {
    if !Zip::from(&data).all(|x| x.is_finite()) {
        return Err(AsinhCodecError::NonFiniteData);
    }

    let mut data = data.into_owned();
    data.mapv_inplace(|x| (x / linear_width).asinh() * linear_width);

    Ok(data)
}

/// Compute `sinh(x/w) * w` over the elements of the input `data` array.
///
/// # Errors
///
/// Errors with
/// - [`AsinhCodecError::NonFiniteData`] if any data element is non-finite
///   (infinite or NaN)
pub fn sinh<T: Float, S: Data<Elem = T>, D: Dimension>(
    data: ArrayBase<S, D>,
    linear_width: T,
) -> Result<Array<T, D>, AsinhCodecError> {
    if !Zip::from(&data).all(|x| x.is_finite()) {
        return Err(AsinhCodecError::NonFiniteData);
    }

    let mut data = data.into_owned();
    data.mapv_inplace(|x| (x / linear_width).sinh() * linear_width);

    Ok(data)
}

#[allow(clippy::needless_pass_by_value)]
/// Compute `sinh(x/w) * w` over the elements of the input `data` array and
/// write them into the `out`put array.
///
/// # Errors
///
/// Errors with
/// - [`AsinhCodecError::NonFiniteData`] if any data element is non-finite
///   (infinite or NaN)
/// - [`AsinhCodecError::MismatchedDecodeIntoArray`] if the `data` array's shape
///   does not match the `out`put array's shape
pub fn sinh_into<T: Float, D: Dimension>(
    data: ArrayView<T, D>,
    mut out: ArrayViewMut<T, D>,
    linear_width: T,
) -> Result<(), AsinhCodecError> {
    if data.shape() != out.shape() {
        return Err(AsinhCodecError::MismatchedDecodeIntoArray {
            source: AnyArrayAssignError::ShapeMismatch {
                src: data.shape().to_vec(),
                dst: out.shape().to_vec(),
            },
        });
    }

    if !Zip::from(&data).all(|x| x.is_finite()) {
        return Err(AsinhCodecError::NonFiniteData);
    }

    // iteration must occur in synchronised (standard) order
    for (d, o) in data.iter().zip(out.iter_mut()) {
        *o = ((*d) / linear_width).sinh() * linear_width;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() -> Result<(), AsinhCodecError> {
        let data = (-1000..1000).map(f64::from).collect::<Vec<_>>();
        let data = Array::from_vec(data);

        let encoded = asinh(data.view(), 1.0)?;

        for (r, e) in data.iter().zip(encoded.iter()) {
            assert_eq!((*r).asinh().to_bits(), (*e).to_bits());
        }

        let decoded = sinh(encoded, 1.0)?;

        for (r, d) in data.iter().zip(decoded.iter()) {
            assert!(((*r) - (*d)).abs() < 1e-12);
        }

        Ok(())
    }

    #[test]
    fn roundtrip_widths() -> Result<(), AsinhCodecError> {
        let data = (-1000..1000).map(f64::from).collect::<Vec<_>>();
        let data = Array::from_vec(data);

        for linear_width in [-100.0, -10.0, -1.0, -0.1, 0.1, 1.0, 10.0, 100.0] {
            let encoded = asinh(data.view(), linear_width)?;
            let decoded = sinh(encoded, linear_width)?;

            for (r, d) in data.iter().zip(decoded.iter()) {
                assert!(((*r) - (*d)).abs() < 1e-12);
            }
        }

        Ok(())
    }
}
