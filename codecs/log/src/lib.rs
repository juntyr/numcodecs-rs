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
//! `ln(x)` codec implementation for the [`numcodecs`] API.

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
/// Log codec which calculates `c = ln(x)` on encoding and `d = exp(c)` on
/// decoding.
///
/// The codec only supports finite positive floating point numbers.
pub struct LogCodec {
    // empty
}

impl Codec for LogCodec {
    type Error = LogCodecError;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        match data {
            AnyCowArray::F32(data) => Ok(AnyArray::F32(ln(data)?)),
            AnyCowArray::F64(data) => Ok(AnyArray::F64(ln(data)?)),
            encoded => Err(LogCodecError::UnsupportedDtype(encoded.dtype())),
        }
    }

    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
        match encoded {
            AnyCowArray::F32(encoded) => Ok(AnyArray::F32(exp(encoded)?)),
            AnyCowArray::F64(encoded) => Ok(AnyArray::F64(exp(encoded)?)),
            encoded => Err(LogCodecError::UnsupportedDtype(encoded.dtype())),
        }
    }

    fn decode_into(
        &self,
        encoded: AnyArrayView,
        decoded: AnyArrayViewMut,
    ) -> Result<(), Self::Error> {
        match (encoded, decoded) {
            (AnyArrayView::F32(encoded), AnyArrayViewMut::F32(decoded)) => {
                exp_into(encoded, decoded)
            }
            (AnyArrayView::F64(encoded), AnyArrayViewMut::F64(decoded)) => {
                exp_into(encoded, decoded)
            }
            (encoded @ (AnyArrayView::F32(_) | AnyArrayView::F64(_)), decoded) => {
                Err(LogCodecError::MismatchedDecodeIntoArray {
                    source: AnyArrayAssignError::DTypeMismatch {
                        src: encoded.dtype(),
                        dst: decoded.dtype(),
                    },
                })
            }
            (encoded, _decoded) => Err(LogCodecError::UnsupportedDtype(encoded.dtype())),
        }
    }
}

impl StaticCodec for LogCodec {
    const CODEC_ID: &'static str = "log";

    type Config<'de> = Self;

    fn from_config(config: Self::Config<'_>) -> Self {
        config
    }

    fn get_config(&self) -> StaticCodecConfig<Self> {
        StaticCodecConfig::from(self)
    }
}

#[derive(Debug, Error)]
/// Errors that may occur when applying the [`LogCodec`].
pub enum LogCodecError {
    /// [`LogCodec`] does not support the dtype
    #[error("Log does not support the dtype {0}")]
    UnsupportedDtype(AnyArrayDType),
    /// [`LogCodec`] does not support non-positive (negative or zero) floating
    /// point data
    #[error("Log does not support non-positive (negative or zero) floating point data")]
    NonPositiveData,
    /// [`LogCodec`] does not support non-finite (infinite or NaN) floating
    /// point data
    #[error("Log does not support non-finite (infinite or NaN) floating point data")]
    NonFiniteData,
    /// [`LogCodec`] cannot decode into the provided array
    #[error("Log cannot decode into the provided array")]
    MismatchedDecodeIntoArray {
        /// The source of the error
        #[from]
        source: AnyArrayAssignError,
    },
}

/// Compute `ln(x)` over the elements of the input `data` array.
///
/// # Errors
///
/// Errors with
/// - [`LogCodecError::NonPositiveData`] if any data element is non-positive
///   (negative or zero)
/// - [`LogCodecError::NonFiniteData`] if any data element is non-finite
///   (infinite or NaN)
pub fn ln<T: Float + Signed, S: Data<Elem = T>, D: Dimension>(
    data: ArrayBase<S, D>,
) -> Result<Array<T, D>, LogCodecError> {
    if !Zip::from(&data).all(T::is_positive) {
        return Err(LogCodecError::NonPositiveData);
    }

    if !Zip::from(&data).all(|x| x.is_finite()) {
        return Err(LogCodecError::NonFiniteData);
    }

    let mut data = data.into_owned();
    data.mapv_inplace(T::ln);

    Ok(data)
}

/// Compute `exp(x)` over the elements of the input `data` array.
///
/// # Errors
///
/// Errors with
/// - [`LogCodecError::NonFiniteData`] if any data element is non-finite
///   (infinite or NaN)
pub fn exp<T: Float, S: Data<Elem = T>, D: Dimension>(
    data: ArrayBase<S, D>,
) -> Result<Array<T, D>, LogCodecError> {
    if !Zip::from(&data).all(|x| x.is_finite()) {
        return Err(LogCodecError::NonFiniteData);
    }

    let mut data = data.into_owned();
    data.mapv_inplace(T::exp);

    Ok(data)
}

#[allow(clippy::needless_pass_by_value)]
/// Compute `exp(x)` over the elements of the input `data` array and write them
/// into the `out`put array.
///
/// # Errors
///
/// Errors with
/// - [`LogCodecError::NonFiniteData`] if any data element is non-finite
///   (infinite or NaN)
/// - [`LogCodecError::MismatchedDecodeIntoArray`] if the `data` array's shape
///   does not match the `out`put array's shape
pub fn exp_into<T: Float, D: Dimension>(
    data: ArrayView<T, D>,
    mut out: ArrayViewMut<T, D>,
) -> Result<(), LogCodecError> {
    if data.shape() != out.shape() {
        return Err(LogCodecError::MismatchedDecodeIntoArray {
            source: AnyArrayAssignError::ShapeMismatch {
                src: data.shape().to_vec(),
                dst: out.shape().to_vec(),
            },
        });
    }

    if !Zip::from(&data).all(|x| x.is_finite()) {
        return Err(LogCodecError::NonFiniteData);
    }

    // iteration must occur in synchronised (standard) order
    for (d, o) in data.iter().zip(out.iter_mut()) {
        *o = T::exp(*d);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() -> Result<(), LogCodecError> {
        let data = (1..1000).map(f64::from).collect::<Vec<_>>();
        let data = Array::from_vec(data);

        let encoded = ln(data.view())?;

        for (r, e) in data.iter().zip(encoded.iter()) {
            assert_eq!((*r).ln().to_bits(), (*e).to_bits());
        }

        let decoded = exp(encoded)?;

        for (r, d) in data.iter().zip(decoded.iter()) {
            assert!(((*r) - (*d)).abs() < 1e-12);
        }

        Ok(())
    }
}
