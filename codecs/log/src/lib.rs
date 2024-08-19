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
//! `ln(x+1)` codec implementation for the [`numcodecs`] API.

use ndarray::{Array, ArrayBase, ArrayView, ArrayViewMut, Data, Dimension};
use numcodecs::{
    AnyArray, AnyArrayAssignError, AnyArrayDType, AnyArrayView,
    AnyArrayViewMut, AnyCowArray, Codec, StaticCodec, StaticCodecConfig,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
/// Log codec which calculates `c = log(1+x)` on encoding and `d = exp(c)-1` on
/// decoding.
///
/// The codec only supports non-negative floating point numbers.
pub struct LogCodec {
    // empty
}

impl Codec for LogCodec {
    type Error = LogCodecError;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        match data {
            AnyCowArray::F32(data) => Ok(AnyArray::F32(ln_1p(data)?)),
            AnyCowArray::F64(data) => Ok(AnyArray::F64(ln_1p(data)?)),
            encoded => Err(LogCodecError::UnsupportedDtype(encoded.dtype())),
        }
    }

    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
        match encoded {
            AnyCowArray::F32(encoded) => Ok(AnyArray::F32(exp_m1(encoded)?)),
            AnyCowArray::F64(encoded) => Ok(AnyArray::F64(exp_m1(encoded)?)),
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
                exp_m1_into(encoded, decoded)
            }
            (AnyArrayView::F64(encoded), AnyArrayViewMut::F64(decoded)) => {
                exp_m1_into(encoded, decoded)
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

    fn from_config<'de>(config: Self::Config<'de>) -> Self {
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
    /// [`LogCodec`] does not support negative floating point data
    #[error("Log does not support negative floating point data")]
    NegativeData,
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

/// Compute `ln(x+1)` over the elements of the input `data` array.
///
/// # Errors
///
/// Errors with
/// - [`LogCodecError::NegativeData`] if any data element is negative
/// - [`LogCodecError::NonFiniteData`] if any data element is non-finite
///   (infinite or NaN)
pub fn ln_1p<T: Float, S: Data<Elem = T>, D: Dimension>(
    data: ArrayBase<S, D>,
) -> Result<Array<T, D>, LogCodecError> {
    if data.iter().copied().any(T::is_negative) {
        return Err(LogCodecError::NegativeData);
    }

    if !data.iter().copied().all(T::is_finite) {
        return Err(LogCodecError::NonFiniteData);
    }

    let mut data = data.into_owned();
    data.mapv_inplace(T::ln_1p);

    Ok(data)
}

/// Compute `exp(x)-1` over the elements of the input `data` array.
///
/// # Errors
///
/// Errors with
/// - [`LogCodecError::NegativeData`] if any data element is negative
/// - [`LogCodecError::NonFiniteData`] if any data element is non-finite
///   (infinite or NaN)
pub fn exp_m1<T: Float, S: Data<Elem = T>, D: Dimension>(
    data: ArrayBase<S, D>,
) -> Result<Array<T, D>, LogCodecError> {
    if data.iter().copied().any(T::is_negative) {
        return Err(LogCodecError::NegativeData);
    }

    if !data.iter().copied().all(T::is_finite) {
        return Err(LogCodecError::NonFiniteData);
    }

    let mut data = data.into_owned();
    data.mapv_inplace(T::exp_m1);

    Ok(data)
}

#[allow(clippy::needless_pass_by_value)]
/// Compute `exp(x)-1` over the elements of the input `data` array and write
/// them into the `out`put array.
///
/// # Errors
///
/// Errors with
/// - [`LogCodecError::NegativeData`] if any data element is negative
/// - [`LogCodecError::NonFiniteData`] if any data element is non-finite
///   (infinite or NaN)
/// - [`LogCodecError::MismatchedDecodeIntoArray`] if the `data` array's shape
///   does not match the `out`put array's shape
pub fn exp_m1_into<T: Float, D: Dimension>(
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

    if data.iter().copied().any(T::is_negative) {
        return Err(LogCodecError::NegativeData);
    }

    if !data.iter().copied().all(T::is_finite) {
        return Err(LogCodecError::NonFiniteData);
    }

    // iteration must occur in synchronised (standard) order
    for (d, o) in data.iter().zip(out.iter_mut()) {
        *o = T::exp_m1(*d);
    }

    Ok(())
}

/// Floating point types.
pub trait Float: Copy {
    /// Returns `ln(self+1)`, the natural logarithm.
    #[must_use]
    fn ln_1p(self) -> Self;

    /// Returns `exp(self)-1`.
    #[must_use]
    fn exp_m1(self) -> Self;

    /// Returns `true` if this number is negative.
    fn is_negative(self) -> bool;

    /// Returns `true` if this number is neither infinite nor NaN.
    fn is_finite(self) -> bool;
}

impl Float for f32 {
    fn ln_1p(self) -> Self {
        self.ln_1p()
    }

    fn exp_m1(self) -> Self {
        self.exp_m1()
    }

    fn is_negative(self) -> bool {
        self.is_sign_negative()
    }

    fn is_finite(self) -> bool {
        self.is_finite()
    }
}

impl Float for f64 {
    fn ln_1p(self) -> Self {
        self.ln_1p()
    }

    fn exp_m1(self) -> Self {
        self.exp_m1()
    }

    fn is_negative(self) -> bool {
        self.is_sign_negative()
    }

    fn is_finite(self) -> bool {
        self.is_finite()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() -> Result<(), LogCodecError> {
        let data = (0..1000).map(|x| x as f64).collect::<Vec<_>>();
        let data = Array::from_vec(data);

        let encoded = ln_1p(data.view())?;

        for (r, e) in data.iter().zip(encoded.iter()) {
            assert_eq!((*r).ln_1p().to_bits(), (*e).to_bits());
        }

        let decoded = exp_m1(encoded)?;

        for (r, d) in data.iter().zip(decoded.iter()) {
            assert!(((*r) - (*d)).abs() < 1e-12);
        }

        Ok(())
    }
}
