//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.87.0-blue
//! [repo]: https://github.com/juntyr/numcodecs-rs
//!
//! [Latest Version]: https://img.shields.io/crates/v/numcodecs-stochastic-rounding
//! [crates.io]: https://crates.io/crates/numcodecs-stochastic-rounding
//!
//! [Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs-stochastic-rounding
//! [docs.rs]: https://docs.rs/numcodecs-stochastic-rounding/
//!
//! [Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/numcodecs-rs/numcodecs_stochastic_rounding
//!
//! Stochastic rounding codec implementation for the [`numcodecs`] API.

use std::borrow::Cow;
use std::hash::{Hash, Hasher};

use ndarray::{Array, ArrayBase, Data, Dimension};
use num_traits::Float;
use numcodecs::{
    AnyArray, AnyArrayAssignError, AnyArrayDType, AnyArrayView, AnyArrayViewMut, AnyCowArray,
    Codec, StaticCodec, StaticCodecConfig, StaticCodecVersion,
};
use rand::{
    SeedableRng,
    distr::{Distribution, Open01},
};
use schemars::{JsonSchema, Schema, SchemaGenerator, json_schema};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;
use wyhash::{WyHash, WyRng};

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
/// Codec that stochastically rounds the data to the nearest multiple of
/// `precision` on encoding and passes through the input unchanged during
/// decoding.
///
/// The nearest representable multiple is chosen such that the absolute
/// difference between the original value and the rounded value do not exceed
/// the precision. Therefore, the rounded value may have a non-zero remainder.
///
/// This codec first hashes the input array data and shape to then `seed` a
/// pseudo-random number generator that is used to sample the stochasticity for
/// rounding. Therefore, passing in the same input with the same `seed` will
/// produce the same stochasticity and thus the same encoded output.
pub struct StochasticRoundingCodec {
    /// The precision of the rounding operation
    pub precision: NonNegative<f64>,
    /// Seed for the random generator
    pub seed: u64,
    /// The codec's encoding format version. Do not provide this parameter explicitly.
    #[serde(default, rename = "_version")]
    pub version: StaticCodecVersion<1, 0, 0>,
}

impl Codec for StochasticRoundingCodec {
    type Error = StochasticRoundingCodecError;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        match data {
            #[expect(clippy::cast_possible_truncation)]
            AnyCowArray::F32(data) => Ok(AnyArray::F32(stochastic_rounding(
                data,
                NonNegative(self.precision.0 as f32),
                self.seed,
            ))),
            AnyCowArray::F64(data) => Ok(AnyArray::F64(stochastic_rounding(
                data,
                self.precision,
                self.seed,
            ))),
            encoded => Err(StochasticRoundingCodecError::UnsupportedDtype(
                encoded.dtype(),
            )),
        }
    }

    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
        match encoded {
            AnyCowArray::F32(encoded) => Ok(AnyArray::F32(encoded.into_owned())),
            AnyCowArray::F64(encoded) => Ok(AnyArray::F64(encoded.into_owned())),
            encoded => Err(StochasticRoundingCodecError::UnsupportedDtype(
                encoded.dtype(),
            )),
        }
    }

    fn decode_into(
        &self,
        encoded: AnyArrayView,
        mut decoded: AnyArrayViewMut,
    ) -> Result<(), Self::Error> {
        if !matches!(encoded.dtype(), AnyArrayDType::F32 | AnyArrayDType::F64) {
            return Err(StochasticRoundingCodecError::UnsupportedDtype(
                encoded.dtype(),
            ));
        }

        Ok(decoded.assign(&encoded)?)
    }
}

impl StaticCodec for StochasticRoundingCodec {
    const CODEC_ID: &'static str = "stochastic-rounding.rs";

    type Config<'de> = Self;

    fn from_config(config: Self::Config<'_>) -> Self {
        config
    }

    fn get_config(&self) -> StaticCodecConfig<'_, Self> {
        StaticCodecConfig::from(self)
    }
}

#[expect(clippy::derive_partial_eq_without_eq)] // floats are not Eq
#[derive(Copy, Clone, PartialEq, PartialOrd, Hash)]
/// Non-negative floating point number
pub struct NonNegative<T: Float>(T);

impl Serialize for NonNegative<f64> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_f64(self.0)
    }
}

impl<'de> Deserialize<'de> for NonNegative<f64> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let x = f64::deserialize(deserializer)?;

        if x >= 0.0 {
            Ok(Self(x))
        } else {
            Err(serde::de::Error::invalid_value(
                serde::de::Unexpected::Float(x),
                &"a non-negative value",
            ))
        }
    }
}

impl JsonSchema for NonNegative<f64> {
    fn schema_name() -> Cow<'static, str> {
        Cow::Borrowed("NonNegativeF64")
    }

    fn schema_id() -> Cow<'static, str> {
        Cow::Borrowed(concat!(module_path!(), "::", "NonNegative<f64>"))
    }

    fn json_schema(_gen: &mut SchemaGenerator) -> Schema {
        json_schema!({
            "type": "number",
            "minimum": 0.0
        })
    }
}

#[derive(Debug, Error)]
/// Errors that may occur when applying the [`StochasticRoundingCodec`].
pub enum StochasticRoundingCodecError {
    /// [`StochasticRoundingCodec`] does not support the dtype
    #[error("StochasticRounding does not support the dtype {0}")]
    UnsupportedDtype(AnyArrayDType),
    /// [`StochasticRoundingCodec`] cannot decode into the provided array
    #[error("StochasticRounding cannot decode into the provided array")]
    MismatchedDecodeIntoArray {
        /// The source of the error
        #[from]
        source: AnyArrayAssignError,
    },
}

/// Stochastically rounds the `data` to the nearest multiple of the `precision`.
///
/// The nearest representable multiple is chosen such that the absolute
/// difference between the original value and the rounded value do not exceed
/// the precision. Therefore, the rounded value may have a non-zero remainder.
///
/// This function first hashes the input array data and shape to then `seed` a
/// pseudo-random number generator that is used to sample the stochasticity for
/// rounding. Therefore, passing in the same input with the same `seed` will
/// produce the same stochasticity and thus the same encoded output.
#[must_use]
pub fn stochastic_rounding<T: FloatExt, S: Data<Elem = T>, D: Dimension>(
    data: ArrayBase<S, D>,
    precision: NonNegative<T>,
    seed: u64,
) -> Array<T, D>
where
    Open01: Distribution<T>,
{
    let mut encoded = data.into_owned();

    if precision.0.is_zero() {
        return encoded;
    }

    let mut hasher = WyHash::with_seed(seed);
    // hashing the shape provides a prefix for the flattened data
    encoded.shape().hash(&mut hasher);
    // the data must be visited in a defined order
    encoded
        .iter()
        .copied()
        .for_each(|x| x.hash_bits(&mut hasher));
    let seed = hasher.finish();

    let mut rng: WyRng = WyRng::seed_from_u64(seed);

    // the data must be visited in a defined order
    for x in &mut encoded {
        if !x.is_finite() {
            continue;
        }

        let remainder = x.rem_euclid(precision.0);

        // compute the nearest multiples of precision based on the remainder
        // correct max 1 ULP rounding errors to ensure that the nearest
        //  multiples are at most precision away from the original value
        let mut lower = *x - remainder;
        if (*x - lower) > precision.0 {
            lower = lower.next_up();
        }
        let mut upper = *x + (precision.0 - remainder);
        if (upper - *x) > precision.0 {
            upper = upper.next_down();
        }

        let threshold = remainder / precision.0;

        let u01: T = Open01.sample(&mut rng);

        // if remainder = 0, U(0, 1) >= 0, so lower (i.e. a) is always picked
        // if threshold = 1/2, U(0, 1) picks lower and upper with equal chance
        *x = if u01 >= threshold { lower } else { upper };
    }

    encoded
}

/// Floating point types
pub trait FloatExt: Float {
    /// -0.5
    const NEG_HALF: Self;

    /// Hash the binary representation of the floating point value
    fn hash_bits<H: Hasher>(self, hasher: &mut H);

    /// Calculates the least nonnegative remainder of self (mod rhs).
    #[must_use]
    fn rem_euclid(self, rhs: Self) -> Self;

    /// Returns the least number greater than `self`.
    #[must_use]
    fn next_up(self) -> Self;

    /// Returns the greatest number less than `self`.
    #[must_use]
    fn next_down(self) -> Self;
}

impl FloatExt for f32 {
    const NEG_HALF: Self = -0.5;

    fn hash_bits<H: Hasher>(self, hasher: &mut H) {
        hasher.write_u32(self.to_bits());
    }

    fn rem_euclid(self, rhs: Self) -> Self {
        Self::rem_euclid(self, rhs)
    }

    fn next_up(self) -> Self {
        Self::next_up(self)
    }

    fn next_down(self) -> Self {
        Self::next_down(self)
    }
}

impl FloatExt for f64 {
    const NEG_HALF: Self = -0.5;

    fn hash_bits<H: Hasher>(self, hasher: &mut H) {
        hasher.write_u64(self.to_bits());
    }

    fn rem_euclid(self, rhs: Self) -> Self {
        Self::rem_euclid(self, rhs)
    }

    fn next_up(self) -> Self {
        Self::next_up(self)
    }

    fn next_down(self) -> Self {
        Self::next_down(self)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{array, linspace};

    use super::*;

    #[test]
    fn round_zero_precision() {
        let data = array![1.1, 2.1];

        let rounded = stochastic_rounding(data.view(), NonNegative(0.0), 42);

        assert_eq!(data, rounded);
    }

    #[test]
    fn round_infinite_precision() {
        let data = array![1.1, 2.1];

        let rounded = stochastic_rounding(data.view(), NonNegative(f64::INFINITY), 42);

        assert_eq!(rounded, array![0.0, 0.0]);
    }

    #[test]
    fn round_minimal_precision() {
        let data = array![0.1, 1.0, 11.0, 21.0];

        assert_eq!(11.0 / f64::MIN_POSITIVE, f64::INFINITY);
        let rounded = stochastic_rounding(data.view(), NonNegative(f64::MIN_POSITIVE), 42);

        assert_eq!(data, rounded);
    }

    #[test]
    fn round_edge_cases() {
        let data = array![
            -f64::NAN,
            -f64::INFINITY,
            -42.0,
            -4.2,
            -0.0,
            0.0,
            4.2,
            42.0,
            f64::INFINITY,
            f64::NAN
        ];
        let precision = 1.0;

        let rounded = stochastic_rounding(data.view(), NonNegative(precision), 42);

        for (d, r) in data.into_iter().zip(rounded) {
            assert!((r - d).abs() <= precision || d.to_bits() == r.to_bits());
        }
    }

    #[test]
    fn round_rounding_errors() {
        let data = Array::from_iter(linspace(-100.0, 100.0, 3741));
        let precision = 0.1;

        let rounded = stochastic_rounding(data.view(), NonNegative(precision), 42);

        for (d, r) in data.into_iter().zip(rounded) {
            assert!((r - d).abs() <= precision);
        }
    }

    #[test]
    fn test_rounding_bug() {
        let data = array![
            -1.23540_f32,
            -1.23539_f32,
            -1.23538_f32,
            -1.23537_f32,
            -1.23536_f32,
            -1.23535_f32,
            -1.23534_f32,
            -1.23533_f32,
            -1.23532_f32,
            -1.23531_f32,
            -1.23530_f32,
            1.23540_f32,
            1.23539_f32,
            1.23538_f32,
            1.23537_f32,
            1.23536_f32,
            1.23535_f32,
            1.23534_f32,
            1.23533_f32,
            1.23532_f32,
            1.23531_f32,
            1.23530_f32,
        ];
        let precision = 0.00018_f32;

        let rounded = stochastic_rounding(data.view(), NonNegative(precision), 42);

        for (d, r) in data.into_iter().zip(rounded) {
            assert!((r - d).abs() <= precision);
        }
    }
}
